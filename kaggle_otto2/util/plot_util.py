from typing import Any, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class PlotUtil:
    @staticmethod
    def plot_importance(
        models: List[Any], feat_columns: List[str], max_num_features: int = 200
    ):
        feature_importance_df = pd.DataFrame()
        for i, model in enumerate(models):
            _df = pd.DataFrame()
            try:
                fe = model.feature_importance()
            except Exception:
                fe = model.feature_importances_

            _df["feature_importance"] = fe
            _df["column"] = feat_columns
            _df["fold"] = i + 1
            feature_importance_df = pd.concat(
                [feature_importance_df, _df], axis=0, ignore_index=True
            )

        order = (
            feature_importance_df.groupby("column")
            .sum()[["feature_importance"]]
            .sort_values("feature_importance", ascending=False)
            .index[:max_num_features]
        )

        fig, ax = plt.subplots(figsize=(12, max(4, len(order) * 0.2)))
        sns.boxenplot(
            data=feature_importance_df,
            y="column",
            x="feature_importance",
            order=order,
            ax=ax,
            palette="viridis",
        )
        fig.tight_layout()
        ax.grid()
        return fig, ax
