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
        time: str = None,
    ) -> pl.DataFrame:
        """
        Perform one-hot encoding on categorical columns of the DataFrame.

        Parameters
        ----------
        df : pl.DataFrame 
            The input DataFrame.
        time : str
            The name of the time column in the dataset, if present

        Returns
        -------
        pl.DataFrame
            The DataFrame with one-hot encoded columns.
        Dict
            A dictionary containing the encoded columns.
        """
        categorical_features = self.categorical_features
        encoded_columns = {}

        if time:
            df = df.sort(time)

        for col in df.select(categorical_features).columns:
            if df[col].dtype == pl.String:
                one_hot = df[col].to_dummies()
                encoded_columns[col] = one_hot.columns
                df = df.hstack(one_hot)
                df = df.drop(col)
        return df, encoded_columns

    def inverse_transform(
        self, 
        df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Reverse one-hot encoding in a Polars DataFrame where dummy columns
        follow the pattern: "<original_column>_<value>".

        Parameters
        ----------
        df : pl.DataFrame
            The input Polars DataFrame with one-hot-encoded columns.

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