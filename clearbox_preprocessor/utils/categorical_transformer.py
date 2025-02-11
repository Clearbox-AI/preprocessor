import polars as pl

class CategoricalTransformer:
    def __init__(
        self,
        preprocessor,
    ):
        """
        """
        self.categorical_features = preprocessor.categorical_features
        self.encoded_columns = {}

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
                self.encoded_columns[col] = one_hot.columns
                df = df.hstack(one_hot)
                df = df.drop(col)
        return df

    def reverse_transform(
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
        df = df.select([col for col in df.columns if not df[col].eq(0).all()])
        
        for col, categories in self.encoded_columns.items():        
            # Filter out columns that are missing from df.columns
            existing_categories = [cname for cname in categories if cname in df.columns]

            # Reconstruct the original categorical column
            if existing_categories:
                reconstructed_expr = pl.when(pl.fold(
                    acc=pl.lit(False),
                    function=lambda acc, x: acc | (x == 1),
                    exprs=[pl.col(cname) for cname in existing_categories]
                )).then(pl.fold(
                    acc=pl.lit(None),
                    function=lambda acc, x: pl.when(x == 1).then(pl.lit(x.name).str.replace(col+"_", "")).otherwise(acc),
                    exprs=[pl.col(cname) for cname in existing_categories]
                )).alias(col)

                # Add the reconstructed column to the DataFrame
                df = df.with_columns(reconstructed_expr)
                df = df.drop(existing_categories)

        return df