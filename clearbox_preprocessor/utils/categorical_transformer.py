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
        self.original_encoded_columns = {}

        # Store encoded columns
        df = df.collect()
        for col in df.select(self.categorical_features).columns:
            if df[col].dtype == pl.String:
                one_hot = df[col].to_dummies()
                self.original_encoded_columns[col] = one_hot.columns

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
        encoded_columns = []

        if time:
            df = df.sort(time)

        for col in df.select(categorical_features).columns:
            if df[col].dtype == pl.String:
                one_hot = df[col].to_dummies()
                encoded_columns[col] = one_hot.columns # Remove if "_enc_" trick is on
                df = df.hstack(one_hot)
                df = df.drop(col)
                # name_mapping = {}
                # for enc_col in one_hot.columns:
                #     name_mapping[enc_col]=enc_col.replace(col, f"{col}_enc", 1)
                #     encoded_columns.append(enc_col)
                # df = df.rename(name_mapping)
        return df, encoded_columns

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
        
        # for col, categories in self.original_encoded_columns.items():        
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




        # # Identify dummy columns and group them by the original column name
        # prefix_map = {} 
        # for col_name in df.columns:
        #     if "_enc_" in col_name:
        #         prefix, _ = col_name.split("_enc_", 1)
        #         prefix_map.setdefault(prefix, []).append(col_name)

        # # For each group (prefix), create a new categorical column based on which dummy col == 1
        # for prefix, dummy_cols in prefix_map.items():
        #     # We'll fold over the dummy columns, picking out the name of the column that is 1
        #     # and stripping off the prefix_enc_ part to get the category.
        #     reconstructed_expr = pl.fold(
        #         acc=pl.lit(None),  # start with None
        #         function=lambda acc, x: pl.when(x == 1)
        #                                 .then(pl.lit(x.name.replace(prefix + "_enc_", "")))
        #                                 .otherwise(acc),
        #         exprs=[pl.col(c) for c in dummy_cols]
        #     ).alias(prefix)

        #     # Add the newly created column to the DataFrame
        #     df = df.with_columns(reconstructed_expr)
        #     # Drop the dummy columns now that we've reconstructed the original column
        #     df = df.drop(dummy_cols)
        # return df
        """
        Reverse one-hot encoding in a Polars DataFrame where dummy columns
        follow the pattern: "original_column_value".

        Example column names: color_red, color_blue, color_green, city_Paris, city_Berlin, ...

        Parameters
        ----------
        df : pl.DataFrame
            The input Polars DataFrame with one-hot-encoded columns.
        categorical_features : list[str]
            A list of original categorical column names (e.g., ['color', 'city']).

        Returns
        -------
        pl.DataFrame
            A new DataFrame with categorical columns reconstructed from the dummy columns.
        """

        # Make a copy so we don't modify the original DataFrame outside the method
        df = df.clone()

        # For each categorical feature, find dummy columns of the form "<feature>_<value>"
        for cat_col in self.categorical_features:
            # Identify all columns for this category
            dummy_cols = [col for col in df.columns if col.startswith(f"{cat_col}_")]

            if not dummy_cols:
                # If no dummy columns match this feature, skip it
                continue

            # We'll fold over these dummy columns, picking which column is 1
            # and extracting the suffix (the encoded 'value') to reconstruct the original category
            reconstructed_expr = pl.fold(
                acc=pl.lit(None),
                function=lambda acc, x: pl.when(x == 1)
                                        .then(pl.lit(x.name.removeprefix(f"{cat_col}_")))
                                        .otherwise(acc),
                exprs=[pl.col(c) for c in dummy_cols]
            ).alias(cat_col)

            # Add the newly created column to the DataFrame
            df = df.with_columns(reconstructed_expr)

            # Drop the dummy columns for this category
            df = df.drop(dummy_cols)

        return df