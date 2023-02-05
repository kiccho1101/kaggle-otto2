from typing import Dict, List, Optional, Tuple

import pandas as pd
import polars as pl


class EvaluateUtil:
    @staticmethod
    def calc_score(
        test_labels: pl.DataFrame,
        pred_dfs: Dict[str, pl.DataFrame],
        topk: int = 20,
        top_clicks: Optional[List[int]] = None,
        top_orders: Optional[List[int]] = None,
        verbose=False,
    ) -> Tuple[float, pd.Series, float]:
        pred_dfs["target_click"] = pred_dfs["target_click"].with_columns(
            [
                pl.col("session").cast(pl.Int32),
                pl.lit(0).cast(pl.UInt8).alias("type"),
            ]
        )
        pred_dfs["target_cart"] = pred_dfs["target_cart"].with_columns(
            [
                pl.col("session").cast(pl.Int32),
                pl.lit(1).cast(pl.UInt8).alias("type"),
            ]
        )
        pred_dfs["target_order"] = pred_dfs["target_order"].with_columns(
            [
                pl.col("session").cast(pl.Int32),
                pl.lit(2).cast(pl.UInt8).alias("type"),
            ]
        )

        eval_df = test_labels.join(
            pl.concat(
                [
                    pred_dfs["target_click"],
                    pred_dfs["target_cart"],
                    pred_dfs["target_order"],
                ]
            ),
            on=["session", "type"],
            how="left",
        )
        null_num = eval_df["y_pred"].is_null().sum()
        null_ratio = null_num / len(eval_df)

        if verbose:
            print("null num:", null_num)
            print("len(eval_df):", len(eval_df))
            print("null ratio:", null_ratio)

        y_pred_is_nulls = eval_df["y_pred"].is_null().to_numpy()

        hits = []
        gt_num = []
        for session, t, ground_truth, y_pred, y_pred_is_null in zip(
            eval_df["session"].to_numpy(),
            eval_df["type"].to_numpy(),
            eval_df["ground_truth"].to_numpy(),
            eval_df["y_pred"].to_numpy(),
            y_pred_is_nulls,
        ):
            if y_pred_is_null:
                y_pred = []
            y_pred = y_pred[:topk]
            # topkより少ない場合は、人気商品で補完する
            if len(y_pred) < topk:
                if t == 0 and top_clicks is not None:
                    y_pred = list(y_pred) + list(top_clicks)[: (topk - len(y_pred))]
                elif top_orders is not None:
                    y_pred = list(y_pred) + list(top_orders)[: (topk - len(y_pred))]
            hits.append(len(set(ground_truth).intersection(set(y_pred))))
            gt_num.append(min(len(ground_truth), topk))
        eval_df = eval_df.with_columns(
            [
                pl.Series("hits", hits),
                pl.Series("gt_num", gt_num),
            ]
        )
        scores = (
            eval_df.to_pandas()
            .groupby("type")
            .apply(lambda df: df["hits"].sum() / df["gt_num"].sum())
        )
        score = 0.1 * scores[0] + 0.3 * scores[1] + 0.6 * scores[2]
        return score, scores, null_ratio
