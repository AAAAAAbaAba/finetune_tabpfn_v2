from __future__ import annotations

import sys
import os
# 添加项目根目录到Python路径
DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(DIR_PATH)

import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
import yaml
from pathlib import Path
from sklearn.datasets import load_iris, load_breast_cancer
import numpy as np

from finetuning_scripts.finetune_tabpfn_main import fine_tune_tabpfn
from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier, TabPFNRegressor

def load_data(
        *,
        data_path: str,
        test_size: float = 0.2,
        random_state: int = 42,
        ):
    data = pd.read_csv(data_path)
    feature_columns = data.columns[1:-1]
    print(f"特征列: {list(feature_columns)}")
    print(f"特征数量: {len(feature_columns)}")
    print(f"数据集形状: {data.shape}")
    print(f"标签列: {data.columns[-1]}")

    X = data[feature_columns].values
    y = data[data.columns[-1]].values
    # 分析y的统计特征
    print(f"目标变量统计信息:")
    print(f"  样本数量: {len(y)}")
    print(f"  唯一值数量: {len(np.unique(y))}")
    print(f"  数据类型: {type(y[0])}")
    
    if task_type in ["binary", "multiclass"]:
        # 分类任务统计
        unique_values, counts = np.unique(y, return_counts=True)
        print(f"  类别分布:")
        for val, count in zip(unique_values, counts):
            percentage = (count / len(y)) * 100
            print(f"    类别 {val}: {count} 个样本 ({percentage:.1f}%)")
    else:
        # 回归任务统计
        print(f"  最小值: {np.min(y):.4f}")
        print(f"  最大值: {np.max(y):.4f}")
        print(f"  平均值: {np.mean(y):.4f}")
        print(f"  标准差: {np.std(y):.4f}")
        print(f"  中位数: {np.median(y):.4f}")
    
    print(f"  缺失值数量: {np.sum(pd.isna(y))}")
    print()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test

def run_modelling(
        *,
        time_limit: int,
        finetuning_config: dict,
        categorical_features_index: list[int] | None,
        device: str,
        seed: int,
        data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        task_type: str,
        finetune_type: str,
        ):
    X_train, X_test, y_train, y_test = data
    
    match task_type:
        case "binary":
            val_metric = "roc_auc"
            tabpfn_model = TabPFNClassifier
            lgbm_model = LGBMClassifier
            test_metric = lambda y_test, y_pred: roc_auc_score(y_test, y_pred[:, 1])
            predict_func = lambda model, X: model.predict_proba(X)
            lower_is_better = False
        case "multiclass":
            val_metric = "log_loss"
            tabpfn_model = TabPFNClassifier
            lgbm_model = LGBMClassifier
            test_metric = log_loss
            predict_func = lambda model, X: model.predict_proba(X)
            lower_is_better = True
        case "regression":
            val_metric = "rmse"
            tabpfn_model = TabPFNRegressor
            lgbm_model = LGBMRegressor
            test_metric = root_mean_squared_error
            predict_func = lambda model, X: model.predict(X)
            lower_is_better = True
        case _:
            raise ValueError(f"Invalid task_type: {task_type}")

    save_path_to_fine_tuned_model = os.path.join(DIR_PATH, f"ckpt/fine_tuned_model_{finetune_type}_{task_type}.ckpt")
    path_to_lora_model = os.path.join(DIR_PATH, f"ckpt/lora_model_{finetune_type}_{task_type}.ckpt")
    path_to_base_model = os.path.join(DIR_PATH, f"ckpt/tabpfn-v2-classifier.ckpt") \
        if task_type == "multiclass" or task_type == "binary"\
        else os.path.join(DIR_PATH, f"ckpt/tabpfn-v2-regressor.ckpt")
    if not os.path.exists(path_to_base_model):
        path_to_base_model = "auto"
    fine_tune_tabpfn(
        path_to_base_model=path_to_base_model,
        save_path_to_fine_tuned_model=save_path_to_fine_tuned_model,
        path_to_lora_model=path_to_lora_model,
        # Finetuning HPs
        time_limit=time_limit,
        finetuning_config=finetuning_config,
        validation_metric=val_metric,
        # Input Data
        X_train=X_train,
        y_train=y_train,
        categorical_features_index=categorical_features_index,
        device=device,
        task_type=task_type,
        # Optional
        show_training_curve=True,
        logger_level=0,
        use_wandb=False,
    )    

    # Run Models
    results = {}
    for model_name, model in [
        ("LGBM", lgbm_model(seed=seed, verbosity=-1)),
        (
            "Default-TabPFN", 
            tabpfn_model(
                model_path=path_to_base_model,
                random_state=seed,
                device=device,
                categorical_features_indices=categorical_features_index,
            )
        ),
        (
            "Finetuned-TabPFN",
            tabpfn_model(
                model_path=save_path_to_fine_tuned_model,
                random_state=seed,
                device=device,
                categorical_features_indices=categorical_features_index,
            )
        ),
    ]:
        model.fit(X_train, y_train)
        results[model_name] = test_metric(y_test, predict_func(model, X_test))

    metric_txt = "↑" if not lower_is_better else "↓"
    report = f""" === Experiment Results for [{task_type} ({metric_txt})] ===
    - LGBM            : {results["LGBM"]:.8f}
    - Default TabPFN  : {results["Default-TabPFN"]:.8f}
    - Finetuned TabPFN: {results["Finetuned-TabPFN"]:.8f}""" + "\n" + \
    f""" === Learning HPs ===
    - Learning Rate: {finetuning_config["learning_rate"]}
    - Batch Size: {finetuning_config["batch_size"]}"""
    report_file_path = os.path.join(DIR_PATH, f"fine_tuning_report/fine_tuning_report.txt")
    with open(report_file_path, "a", encoding="utf-8") as f:
        f.write(report + "\n\n\n")
    print(report)


if __name__ == '__main__':
    # Load data
    task_type = "regression"
    finetune_type = "lora-attn"
    if task_type == "regression":
        data_path = 'D:/0 Program/TabPFN/非时序数据集/openml/cars_44994.csv'
        data = load_data(data_path=data_path)
    else:
        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        data = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

    with (Path(__file__).parent / "toy_example/finetuning_hps.yaml").open("r") as file:
        finetuning_config = yaml.safe_load(file)

    run_modelling(
        time_limit=300,
        finetuning_config=finetuning_config,
        categorical_features_index=None,
        device="cuda",
        seed=42,
        data=data,
        task_type=task_type,
        finetune_type=finetune_type,
    )
