
import numpy as np
import gc

class Experiment:
    def __init__(self, data_processor, methods):
        self.data_processor = data_processor
        self.methods = methods  # List of models to train and evaluate
        # self.n_train_tasks = len(data_processor.train_tasks)
        # self.n_samples_per_train_tasks = tuple([len(data_processor.train_tasks[task]) for task in data_processor.train_tasks])
        # self.n_test_tasks = len(data_processor.test_tasks)
        self.results = {method.name: [] for method in methods} 
        self.all_results = []  # Store all results as a list of dicts
  
    
    def _run_incr(self, annot, k):
        target = {}
        for i, (train_tasks, test_tasks, train_tasks_name, test_tasks_name) in enumerate(self.data_processor.leave_percent_tasks_out_cv()):
            print("____________TRAIN LEAVE OUT ON ", i + 1, "____________")
            print(f"Train tasks {train_tasks_name}")
            print(f"Test tasks {test_tasks_name}")
            n_train_tasks = len(train_tasks)
            n_samples_per_train_tasks = tuple([len(train_tasks[task]) for task in train_tasks])
        
            for _, n_tasks in enumerate(np.arange(2, n_train_tasks + 1)):
                print(f"### TASK: {n_tasks} ###")
                params = {
                    'n_samples_per_task': n_samples_per_train_tasks[0:n_tasks],
                }

                selected_train_tasks = {k: v for k, v in list(train_tasks.items())[:n_tasks]}
                selected_test_tasks = {k: v for k, v in list(test_tasks.items())[:n_tasks]}

                X_train, y_train, columns = self.data_processor.get_xy_split(selected_train_tasks)
                X_test, y_test, _ = self.data_processor.get_xy_split(selected_test_tasks)            

                for method in self.methods:
                    print(f"Method name: {method.name}")
                    method.fit(X_train, y_train, params)
                    loss = method.evaluate(X_test, y_test)
                    self.results[method.name].append(loss)

                    result_entry = {
                        "leave_out_task": test_tasks_name[0],  # Assumes one left-out task
                        "n_tasks": n_tasks,
                        "method": method.name,
                        "loss": loss
                    }

                    try:
                        result_entry["selected_features"] = list(columns[method.selected_features])
                        print(f"Selected columns {columns[method.selected_features]}")
                    except Exception:
                        result_entry["selected_features"] = None
                        print("_____")
                    print(f'| Loss: {loss:.6f} |')
                    print("_____")

                    self.all_results.append(result_entry)

            print(f"EXPERIMENT RESULTS")
            for method in self.methods:
                mean_mse = np.mean(self.results[method.name])
                std_mse = np.std(self.results[method.name])
                print(f"  {method.name}: {mean_mse:.6f} ± {std_mse:.6f}") 

                self.results[method.name] = []

        # Save results to CSV
        self._save_results(annot)
        return self

    def _run_once(self, annot, k=3):
        """
        Run training ONCE on top-k tasks (by data size), then test on all remaining tasks.
        Evaluate and record both training and test loss.

        Additionally, for each method, save per-sample predictions and residuals
        along with task and split (train/test) info.
        """
        import pandas as pd
        import os

        # Get top-k tasks for training and remaining tasks for testing
        train_tasks, test_tasks_list, train_tasks_name, test_tasks_names = self.data_processor.get_top_k_split(k=k)

        print("____________ TRAIN ONCE ON TOP-K TASKS ____________")
        print(f"Train tasks: {train_tasks_name}")
        print(f"Test tasks: {test_tasks_names}")

        # Prepare training data once
        X_train, y_train, columns = self.data_processor.get_xy_split(train_tasks)

        # Store per-sample results
        per_sample_records = []

        for method in self.methods:
            print(f"Training method: {method.name}")

            params = {
                'n_samples_per_task': tuple(len(train_tasks[task]) for task in train_tasks),
            }

            method.fit(X_train, y_train, params)

            # --- TRAINING EVALUATION ---
        
            residuals_train =  method.cal_residuals_list(X_train, y_train)
            print(residuals_train.shape)
            predictions_train = method.predict(X_train)

            # Collect per-sample results for training
            offset = 0
            for task_name, task_df in train_tasks.items():
                n_samples = len(task_df)
                for i in range(n_samples):
                    pred = float(predictions_train[offset + i])
                    true = float(y_train[offset + i])
                    res = float(residuals_train[offset + i])
                    per_sample_records.append({
                        "method": method.name,
                        "task": task_name,
                        "split": "train",
                        "sample_idx": i,
                        "prediction": pred,
                        "true_label": true,
                        "residual": res,
                    })
                offset += n_samples

            loss_train = method.evaluate(X_train, y_train)

            try:
                selected_feats = list(columns[method.selected_features])
                print(f"Selected columns: {selected_feats}")
            except Exception:
                selected_feats = None

            print(f'Training Loss: {loss_train:.6f}')

            # --- TESTING EVALUATION ---
            for i, test_task_dict in enumerate(test_tasks_list):
                test_task_name = list(test_task_dict.keys())[0]
                print(f"Testing on task {i+1}: {test_task_name}")

                X_test, y_test, _ = self.data_processor.get_xy_split(test_task_dict)

                predictions_test = method.predict(X_test)
                residuals_test = method.cal_residuals_list(X_test, y_test)

                # Collect per-sample results for testing
                for i in range(len(y_test)):
                    per_sample_records.append({
                        "method": method.name,
                        "task": test_task_name,
                        "split": "test",
                        "sample_idx": i,
                        "prediction": float(predictions_test[i]),
                        "true_label": float(y_test[i]),
                        "residual": float(residuals_test[i]),
                        # "loss": float(method.evaluate(X_test, y_test)),
                        # "loss_2": np.sqrt(np.mean((method.model.predict(X_test) - y_test)**2))
                        # # "loss_2" : np.sqrt(np.mean(method.model.predict(X_test), y_test)))
                    })

                loss_test = method.evaluate(X_test, y_test)

                try:
                    selected_feats = list(columns[method.selected_features])
                except Exception:
                    selected_feats = None

                self.all_results.append({
                    "leave_out_task": test_task_name,
                    "n_tasks": len(train_tasks),
                    "method": method.name,
                    "loss": loss_test,
                    "selected_features": selected_feats,
                })

                print(f'Test Loss on {test_task_name}: {loss_test:.6f}')

            # Store training summary once per method
            self.all_results.append({
                "leave_out_task": "training_set",
                "n_tasks": len(train_tasks),
                "method": method.name,
                "loss": loss_train,
                "split": "train"
            })

            print("-----")

        # Save per-sample predictions and residuals
        output_dir = os.path.join(os.path.dirname(__file__), 'per_sample_results')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{annot}.csv')
        pd.DataFrame(per_sample_records).to_csv(output_file, index=False)
        print(f"Saved per-sample results to {output_file}")

        self._save_results(annot)
        return self

    def _run(self, mode, annot=None, k=40):
        """
        Run the experiment based on the specified mode.
        
        Parameters:
        - mode: str, either 'train' or 'test'
        - annot: Optional annotation for identifying this run
        - k: Number of top tasks to use for training
        """
        if mode == 0:
            self._run_once(annot, k=k)
        elif mode == 1:
            self._run_incr(annot, k=k)
        else:
            raise ValueError("Invalid mode. Use 'train' or 'test'.")

    def _save_results(self, annot):
        import pandas as pd
        output_file = f"logs/{annot}.csv"
        df = pd.DataFrame(self.all_results)
        df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to {output_file}")

    def run_experiment(self, mode, annot=None, k=40):
        """
        Run the full experiment with training on top-k tasks and per-task testing.
        Print average results for each method on both train and test splits.

        Parameters:
        - annot: Optional annotation for identifying this run
        - k: Number of top tasks to use for training
        """
        self._run(mode=mode, annot=annot, k=k)

        if annot is not None:
            print(f"#__________ FINAL RESULTS FOR EXPERIMENT: {annot} ___________#")

        print(f"Experiment setup - Train on top {k} tasks")
        print(f"All tasks: {list(self.data_processor.tasks.keys())}")

        from collections import defaultdict
        import numpy as np

        # Separate results by method and split
        results_by_method_split = defaultdict(lambda: {"train": [], "test": []})
        for result in self.all_results:
            method = result["method"]
            split = result.get("split", "test")  # Default to test if not specified
            results_by_method_split[method][split].append(result["loss"])

        # Print results
        for method in self.methods:
            method_name = method.name
            train_losses = results_by_method_split[method_name]["train"]
            test_losses = results_by_method_split[method_name]["test"]

            if train_losses:
                mean_train = np.mean(train_losses)
                std_train = np.std(train_losses)
                print(f"  {method_name} [Train]: {mean_train:.6f} ± {std_train:.6f}")
            else:
                print(f"  {method_name} [Train]: No results available.")

            if test_losses:
                mean_test = np.mean(test_losses)
                std_test = np.std(test_losses)
                print(f"  {method_name} [Test] : {mean_test:.6f} ± {std_test:.6f}")
            else:
                print(f"  {method_name} [Test] : No results available.")
