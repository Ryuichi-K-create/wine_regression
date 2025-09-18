# src/models/svr.py
from __future__ import annotations
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any, Optional


class SVRRegressor:
    """サポートベクター回帰モデル"""
    
    def __init__(self, 
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 epsilon: float = 0.1,
                 gamma: str = 'scale',
                 grid_search: bool = True,
                 cv: int = 5,
                 random_state: int = 42):
        """
        Args:
            kernel: カーネル関数 ('rbf', 'linear', 'poly', 'sigmoid')
            C: 正則化パラメータ
            epsilon: epsilon-tube内の誤差を無視
            gamma: カーネル係数
            grid_search: グリッドサーチでハイパーパラメータ調整するか
            cv: クロスバリデーションの分割数
            random_state: 乱数シード
        """
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.grid_search = grid_search
        self.cv = cv
        self.random_state = random_state
        
        self.model = None
        self.best_params_ = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        モデルを学習
        
        Args:
            X_train: 訓練データの特徴量
            y_train: 訓練データの目的変数
            X_val: 検証データの特徴量（使用されないが互換性のため）
            y_val: 検証データの目的変数（使用されないが互換性のため）
            
        Returns:
            学習履歴（SVRの場合は空のdict）
        """
        if self.grid_search:
            # グリッドサーチでハイパーパラメータ調整
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'epsilon': [0.01, 0.1, 0.2, 0.5],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
            }
            
            svr = SVR(kernel=self.kernel)
            grid_search = GridSearchCV(
                svr, param_grid, 
                cv=self.cv, 
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params_ = grid_search.best_params_
            
            print(f"[SVR] Best parameters: {self.best_params_}")
            print(f"[SVR] Best CV score: {-grid_search.best_score_:.4f}")
        else:
            # 指定されたパラメータで学習
            self.model = SVR(
                kernel=self.kernel,
                C=self.C,
                epsilon=self.epsilon,
                gamma=self.gamma
            )
            self.model.fit(X_train, y_train)
            
        return {}  # SVRには学習履歴がないので空のdict
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測を実行"""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        """モデルを保存"""
        import joblib
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        joblib.dump(self.model, path)
    
    def load(self, path: str) -> None:
        """モデルを読み込み"""
        import joblib
        self.model = joblib.load(path)
    
    @property
    def n_params(self) -> int:
        """パラメータ数（SVRの場合はサポートベクタ数）"""
        if self.model is None:
            return 0
        return len(self.model.support_)
