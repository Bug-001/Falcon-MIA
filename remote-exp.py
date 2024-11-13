import yaml
import threading
import queue
import paramiko
import time
import os
import sys
import logging
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import PurePosixPath
from queue import Queue

class OutputMonitor:
    def __init__(self, stdout, thread_name="stdout-monitor"):
        self.stdout = stdout
        self.thread_name = thread_name
        self._stop_event = threading.Event()
        self.queue = Queue()
        self.monitor_thread = None

    def start_monitoring(self):
        self.monitor_thread = threading.Thread(
            target=self._monitor_output,
            name=self.thread_name,
            daemon=True
        )
        self.monitor_thread.start()

    def stop_monitoring(self):
        self._stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_output(self):
        while not self._stop_event.is_set():
            line = self.stdout.readline()
            if line:
                # Put the line in queue and print it
                self.queue.put(line)
                sys.stdout.write(f"[{self.thread_name}] {line}")
                sys.stdout.flush()
            else:
                time.sleep(0.1)  # Prevent busy-waiting

class ExperimentThread(threading.Thread):
    def __init__(self, thread_id, model_queue, ssh_host, ssh_user):
        super().__init__()
        self.thread_id = thread_id
        self.model_queue = model_queue
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        
        # Create output/log directory if it doesn't exist
        os.makedirs("output/log", exist_ok=True)
        os.makedirs("output/server-config", exist_ok=True)
        os.makedirs("output/param-config", exist_ok=True)
        
    def create_param_configs(self, base_param_config):
        """
        Create multiple parameter configurations by expanding parallel parameters
        Returns a list of (param_file_path, param_config) tuples
        """
        configs = []
        
        # Deep copy the base config to avoid modifying the original
        param_config = deepcopy(base_param_config)
        
        # Create individual configs for each parameter value
        # parallel_params = param_config['params'][0]['config']['dataset']
        # for i, param_value in enumerate(parallel_params):
        #     new_config = deepcopy(param_config)
        #     new_config['params'][0]['config']['dataset'] = [param_value]  # Single value in list
            
        #     # Create parameter file name based on the value
        #     param_file = f"output/param-config/param-{self.thread_id}-{param_value['value']}.yaml"
            
        #     # Save the configuration
        #     with open(param_file, 'w') as f:
        #         yaml.dump(new_config, f)
            
        #     configs.append((param_file, new_config))

        # XXX: No-parallel
        param_file = f"output/param-config/param-{self.thread_id}.yaml"
        configs.append((param_file, param_config))
        with open(param_file, 'w') as f:
            yaml.dump(param_config, f)
        
        return configs

    def update_server_config(self, model_name, num_gpus):
        # Create or update server config file
        server_file = f"output/server-config/server-{self.thread_id}.yaml"
        if not os.path.exists(server_file):
            with open("server.yaml", 'r') as f:
                config = yaml.safe_load(f)
        else:
            with open(server_file, 'r') as f:
                config = yaml.safe_load(f)
        
        config['model-tag'] = model_name
        config['host'] = "174.0.241.6"
        config['port'] = 8000 + self.thread_id
        config['tensor-parallel-size'] = num_gpus
        config['gpu-memory-utilization'] = 0.99
        
        with open(server_file, 'w') as f:
            yaml.dump(config, f)
        return server_file
    
    def update_params_config(self, model_name):
        with open("params.yaml", 'r') as f:
            param_config = yaml.safe_load(f)
        
        # Update name_prefix with model name (part after slash)
        param_config['name_prefix'] = model_name.split('/')[-1]
        # XXX: param_config['params'][2]['config'] is query_config
        param_config['params'][2]['config']['model'] = [f"../../../output/server-config/server-{self.thread_id}.yaml"]
        
        return param_config
    
    def create_remote_directory(self, sftp, remote_path):
        """Recursively create remote directory if it doesn't exist"""
        if remote_path == '/':
            return
        try:
            sftp.stat(remote_path)
        except IOError:
            # Convert the path to PurePosixPath to handle the path in Unix style
            parent = str(PurePosixPath(remote_path).parent)
            self.create_remote_directory(sftp, parent)
            try:
                sftp.mkdir(remote_path)
            except IOError as e:
                # If multiple threads try to create the same directory,
                # one might fail because another already created it
                if "already exists" not in str(e).lower():
                    raise
    
    def sync_file_to_remote(self, ssh_client, local_file, remote_base_path="/home/export/base/sc100085/sc100085/online1/remote/better-MIA/"):
        """
        Sync local file to remote server, creating directories as needed
        
        Args:
            ssh_client: paramiko SSH client
            local_file: local file path to sync
            remote_base_path: base path on remote server
        """
        sftp = ssh_client.open_sftp()
        try:
            # Construct remote path
            remote_file = str(PurePosixPath(remote_base_path) / local_file)
            remote_dir = str(PurePosixPath(remote_file).parent)
            
            # Create remote directory structure if needed
            self.create_remote_directory(sftp, remote_dir)
            
            # Upload the file
            sftp.put(local_file, remote_file)
            print(f"Successfully uploaded {local_file} to {remote_file}")
            
        except Exception as e:
            print(f"Error syncing file {local_file}: {str(e)}")
            raise
        finally:
            sftp.close()
    
    def wait_for_server_ready(self, stdout, stderr, target_string="Uvicorn running on socket", timeout_minutes=600):
            # Set timeout for the channel
            start_time = time.time()
            timeout = timeout_minutes * 60  # convert to seconds
            
            while time.time() - start_time < timeout:
                # Check stdout
                try:
                    line = stdout.readline().strip()
                    if line:
                        print(f"Thread {self.thread_id} - Server output: {line}")
                        if target_string in line:
                            return True
                except IOError:
                    pass
                    
                # Check stderr
                try:
                    line = stderr.readline().strip()
                    if line:
                        print(f"Thread {self.thread_id} - Server error: {line}")
                        if target_string in line:
                            return True
                except IOError:
                    pass
                    
                time.sleep(0.1)  # Short sleep to prevent CPU spinning
                
            raise TimeoutError(f"Server failed to start within {timeout_minutes} minutes")
    
    def setup_logging(self, model_name):
        # Create a unique log file name based on timestamp, model name and thread id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short_name = model_name.split('/')[-1]
        log_file = f"output/log/experiment_{timestamp}_{model_short_name}_thread{self.thread_id}.log"
        
        # Create file handler and set level to INFO
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Create logger
        logger = logging.getLogger(f"experiment_thread_{self.thread_id}")
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        
        return logger, log_file
    
    def run_experiments(self, param_configs, log_file):
        """
        Run multiple experiments in parallel using subprocess
        """
        processes = []
        for param_file, _ in param_configs:
            # Create command
            cmd = f"python run.py --params {param_file}"
            
            # Start process and redirect output to log file
            with open(log_file, 'a') as f:
                process = subprocess.Popen(
                    cmd.split(),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                processes.append(process)
        
        # Wait for all processes to complete
        for process in processes:
            process.wait()

    def get_ssh_client(self):
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(self.ssh_host, username=self.ssh_user)
        transport = ssh_client.get_transport()
        transport.set_keepalive(1)
        return ssh_client

    def run(self):
        ssh_client = self.get_ssh_client()
        
        while True:
            try:
                # Get model info from queue
                model_info = self.model_queue.get_nowait()
                if isinstance(model_info, tuple):
                    model_name, num_gpus = model_info
                else:
                    model_name, num_gpus = model_info, 1
                
            except queue.Empty:
                break
                
            try:
                print(f"Thread {self.thread_id} starting experiment with model: {model_name} (using {num_gpus} GPUs)")
                
                # Setup logging
                logger, log_file = self.setup_logging(model_name)
                logger.info(f"Starting experiment with model: {model_name} using {num_gpus} GPUs")
                
                # Update and sync server config
                server_file = self.update_server_config(model_name, num_gpus)
                self.sync_file_to_remote(ssh_client, server_file)
                
                # Update params and create multiple configs for parallel execution
                base_param_config = self.update_params_config(model_name)
                param_configs = self.create_param_configs(base_param_config)
                
                # Start LLM server
                command = f"cd online1/remote/better-MIA && srun -p q_amd_gpu_nvidia_1 --gres=gpu:{num_gpus} python llm/server.py -c output/server-config/server-{self.thread_id}.yaml"
                _, stdout, stderr = ssh_client.exec_command(command)
                
                # XXX
                # self.run_experiments(param_configs, log_file)

                # Wait for server to be ready
                if self.wait_for_server_ready(stdout, stderr):
                    print(f"Thread {self.thread_id} - Server is ready!")

                    # Run all parameter configurations in parallel
                    monitor = OutputMonitor(stderr)
                    monitor.start_monitoring()
                    self.run_experiments(param_configs, log_file)
                    
                logger.info("Experiment completed successfully")
                
            except KeyboardInterrupt:
                logger.info("Experiment interrupted by user")
                break
            
            except Exception as e:
                logger.error(f"Error in experiment: {str(e)}")
                raise
            
            finally:
                # Cancel all servers
                print("Cancelling all servers")
                ssh_client.exec_command("scancel -u sc100085")
                ssh_client.close()
                self.model_queue.task_done()
                monitor.stop_monitoring()
        
        ssh_client.close()

def main():
    # List of models to experiment with, each entry can be either:
    # - just the model name (string) for single GPU usage
    # - a tuple of (model_name, num_gpus) for multi-GPU usage
    models = [
        "meta-llama/Meta-Llama-3-8B-Instruct", # Uses 1 GPU by default
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        ("Qwen/Qwen2.5-14B-Instruct", 2),
        ("mistralai/Mistral-7B-Instruct-v0.2", 2),
        # Add more models as needed
    ]
    
    # Create a queue and fill it with models
    model_queue = queue.Queue()
    for model in models:
        model_queue.put(model)
    
    # SSH connection details
    ssh_config = {
        "host": "174.0.250.88",
        "user": "sc100085",
    }
    
    # Create and start threads (maximum 8)
    threads = []
    num_threads = min(4, len(models))
    
    for thread_id in range(num_threads):
        thread = ExperimentThread(
            thread_id=thread_id,
            model_queue=model_queue,
            ssh_host=ssh_config["host"],
            ssh_user=ssh_config["user"],
        )
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    try:
        for thread in threads:
            thread.join()
    finally:
        print("Cancelling all servers")
        ssh_client = thread.get_ssh_client()
        ssh_client.exec_command("scancel -u sc100085")

    print("All experiments completed!")

if __name__ == "__main__":
    main()