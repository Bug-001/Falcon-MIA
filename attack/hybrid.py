class HybridModelTrainer(BaseTrainer):
    def __init__(self, config: Dict[str, Any], logger):
        super().__init__(config, logger)
        self.hidden_size = config.get('hidden_size', 10)
    
    def get_model(self) -> nn.Module:
        class HybridModel(nn.Module):
            def __init__(self, hidden_size: int):
                super().__init__()
                self.fc1 = nn.Linear(2, hidden_size)
                self.fc2 = nn.Linear(hidden_size, 1)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.sigmoid(self.fc2(x))
                return x
                
        return HybridModel(self.hidden_size)
    
    def get_criterion(self) -> nn.Module:
        return nn.BCELoss()

class HybridAttack(ICLAttackStrategy):
    def __init__(self, attack_config: Dict[str, Any]):
        super().__init__(attack_config)

        # If the random seed is not given in the config, this will ensure the random seed is totally the same for dataloaders of both sub-attacks
        attack_config['random_seed'] = self.random_seed

        self.brainwash_attack = BrainwashAttack(attack_config)
        self.repeat_attack = RepeatAttack(attack_config)
        self.trainer = HybridModelTrainer(attack_config, self.logger)
        self.model = None

    def plot_loss_curve(self, losses):
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        self.logger.savefig('hybrid_training_loss_curve.png')
        plt.close()

    def prepare(self, data_config: Dict[str, Any]):
        super().prepare(data_config)
        # WARNING: Be cautious for the potential data race, if any
        self.brainwash_attack.prepare(data_config, self.data_loader)
        self.repeat_attack.prepare(data_config, self.data_loader)

    # @ICLAttackStrategy.cache_results
    def attack(self, model: 'ModelInterface'):
        # 执行Brainwash和Repeat攻击
        self.brainwash_attack.attack(model)
        self.repeat_attack.attack(model)
        
        # 准备数据
        brainwash_data = [score for _, _, score in self.brainwash_attack.results]
        repeat_data = [score for _, _, score in self.repeat_attack.results]

        # 归一化
        brainwash_data = MinMaxScaler().fit_transform(np.array(brainwash_data).reshape(-1, 1)).flatten()
        repeat_data = MinMaxScaler().fit_transform(np.array(repeat_data).reshape(-1, 1)).flatten()
        data = list(zip(brainwash_data, repeat_data))
        labels = [float(is_member) for _, is_member, _ in self.brainwash_attack.results]

        # 绘制散点图
        self.plot_attack_scores(brainwash_data, repeat_data, np.array(labels))

        # 划分训练集和测试集
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=0.3, random_state=self.random_seed
        )

        # 将数据转换为PyTorch张量并移到GPU
        train_data = torch.tensor(train_data, dtype=torch.float32).to(self.device)
        train_labels = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1).to(self.device)
        test_data = torch.tensor(test_data, dtype=torch.float32).to(self.device)

        # 训练模型
        logger.info("Starting model training...")
        training_results = self.trainer.train(train_data, train_labels)
        self.model = training_results['model']
        logger.info("Model training completed.")

        # 进行混合攻击预测（使用测试集）
        self.model.eval()
        with torch.no_grad():
            for data, is_member in zip(test_data, test_labels):
                pred = self.model(data.unsqueeze(0)).item()
                pred_member = pred > 0.5
                self.results.append((pred_member, is_member, pred))

                logger.info(f"Hybrid Attack Result:")
                logger.info(f"Brainwash score: {data[0].item()}")
                logger.info(f"Repeat score: {data[1].item()}")
                logger.info(f"Is member: {is_member}, Predicted member: {pred_member}")
                logger.info("-" * 50)

    def plot_attack_scores(self, brainwash_scores, repeat_scores, labels):
        plt.figure(figsize=(10, 8))
        
        # 使用MinMaxScaler归一化scores，使其更容易可视化
        scaler = MinMaxScaler()
        brainwash_scores_scaled = scaler.fit_transform(np.array(brainwash_scores).reshape(-1, 1)).flatten()
        repeat_scores_scaled = scaler.fit_transform(np.array(repeat_scores).reshape(-1, 1)).flatten()

        # 为成员和非成员样本创建不同的散点图
        plt.scatter(brainwash_scores_scaled[labels==1], repeat_scores_scaled[labels==1], 
                    c='red', label='Member', alpha=0.6)
        plt.scatter(brainwash_scores_scaled[labels==0], repeat_scores_scaled[labels==0], 
                    c='blue', label='Non-member', alpha=0.6)

        plt.xlabel('Brainwash Attack Score (Normalized)')
        plt.ylabel('Repeat Attack Score (Normalized)')
        plt.title('Hybrid Attack: Brainwash vs Repeat Scores')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # 保存图像
        self.logger.savefig('hybrid_attack_scores.png')
        plt.close()

        logger.info("Hybrid attack scores plot saved as 'hybrid_attack_scores.png'")

    def evaluate(self) -> Dict[str, float]:
        predictions = [bool(pred) for pred, _, _ in self.results]
        ground_truth = [bool(truth) for _, truth, _ in self.results]
        scores = [score for _, _, score in self.results]
        
        metrics = EvaluationMetrics.calculate_advantage(predictions, ground_truth)
        
        fpr, tpr, roc_auc = EvaluationMetrics.calculate_roc_auc(ground_truth, scores)
        log_fpr, log_tpr, log_auc = EvaluationMetrics.calculate_log_roc_auc(ground_truth, scores)
        
        metrics['auc'] = roc_auc
        metrics['log_auc'] = log_auc

        if self.attack_config.get('plot_roc', False):
            EvaluationMetrics.plot_roc(fpr, tpr, roc_auc, 'hybrid_roc_curve.png')
            EvaluationMetrics.plot_log_roc(log_fpr, log_tpr, log_auc, 'hybrid_log_roc_curve.png')

        print(f"Repeat: {self.repeat_attack.evaluate()}")
        print(f"Brainwash: {self.brainwash_attack.evaluate()}")

        # self.logger.save_json('evaluation.json', metrics)

        return metrics