import polars as pl

class CategoricalTransformer:
    def __init__(
        self,
        df,
        preprocessor,
    ):
        """
        """
        self.categorical_features = preprocessor.categorical_features
        self.encoded_columns = {}

        # Store encoded columns
        for col in df.select(self.categorical_features).columns:
            if df[col].dtype == pl.String:
                one_hot = df[col].to_dummies()
                self.encoded_columns[col] = one_hot.columns

    def transform(
        self, 
        df: pl.DataFrame,
        time = None,
    ) -> pl.DataFrame:
        """
        Perform one-hot encoding on categorical columns of the DataFrame.

        Parameters:
        df (pl.DataFrame): The input DataFrame.

        Returns:
        pl.DataFrame: The DataFrame with one-hot encoded columns.
        """
        categorical_features = self.categorical_features
        self.encoded_columns = {}

        if time:
            df = df.sort(time)

        for col in df.select(categorical_features).columns:
            if df[col].dtype == pl.String:
                one_hot = df[col].to_dummies()
                # self.encoded_columns[col] = one_hot.columns
                df = df.hstack(one_hot)
                df = df.drop(col)
                name_mapping = {}
                for enc_col in one_hot.columns:
                    name_mapping[enc_col]=enc_col.replace(col, f"{col}_enc", 1)
                df.rename(name_mapping)
        return df

    def inverse_transform(
        self, 
        df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Perform the inverse transformation on one-hot encoded categorical columns.

        Parameters:
        df (pl.DataFrame): The input DataFrame with one-hot encoded columns.

        Returns:
        pl.DataFrame: The reversed original DataFrame.
        """
        # Drop all zeros columns
        # df_clean = df.select([col for col in df.columns if not (df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32] and df[col].eq(0).all())])
        
        # for col, categories in self.encoded_columns.items():        
        #     # Filter out columns that are missing from df.columns
        #     existing_categories = [cname for cname in categories if cname in df.columns]

        #     # Reconstruct the original categorical column
        #     if existing_categories:
        #         reconstructed_expr = pl.when(pl.fold(
        #             acc=pl.lit(False),
        #             function=lambda acc, x: acc | (x == 1),
        #             exprs=[pl.col(cname) for cname in existing_categories]
        #         )).then(pl.fold(
        #             acc=pl.lit(None),
        #             function=lambda acc, x: pl.when(x == 1).then(pl.lit(x.name).str.replace(col+"_", "")).otherwise(acc),
        #             exprs=[pl.col(cname) for cname in existing_categories]
        #         )).alias(col)

        #         # Add the reconstructed column to the DataFrame
        #         df = df.with_columns(reconstructed_expr)
        #         df = df.drop(existing_categories)




        # 1. Identify dummy columns and group them by the original column name
        prefix_map = {}  # e.g. {"color": ["color_enc_red", "color_enc_green"], "city": [...], ...}
        for col_name in df.columns:
            if "_enc_" in col_name:
                prefix, _ = col_name.split("_enc_", 1)
                prefix_map.setdefault(prefix, []).append(col_name)

        # 2. For each group (prefix), create a new categorical column based on which dummy col == 1
        for prefix, dummy_cols in prefix_map.items():
            # We'll fold over the dummy columns, picking out the name of the column that is 1
            # and stripping off the prefix_enc_ part to get the category.
            reconstructed_expr = pl.fold(
                acc=pl.lit(None),  # start with None
                function=lambda acc, x: pl.when(x == 1)
                                        .then(pl.lit(x.name.replace(prefix + "_enc_", "")))
                                        .otherwise(acc),
                exprs=[pl.col(c) for c in dummy_cols]
            ).alias(prefix)

            # Add the newly created column to the DataFrame
            df = df.with_columns(reconstructed_expr)
            # Drop the dummy columns now that we've reconstructed the original column
            df = df.drop(dummy_cols)
        return df