import polars as pl
import dateinfer
from dateutil.parser import parse

class DatetimeTransformer():
    def __init__(
        self,
        preprocessor
        ):
        self.datetime_formats = {}
        self.dividers = []
        self.datetime_features = preprocessor.time_featuers if hasattr(preprocessor, "time_featuers") else tuple()
        self.scaling = preprocessor.scaling
 
    @staticmethod
    def _is_date_string(value):
        try:
            parse(value)
            return True
        except ValueError:
            return False

    def _try_convert_to_datetime(self, df, col):
        datetime_formats_to_dtype = {
            "%Y-%m-%d %H:%M:%S%.f": pl.Datetime,
            "%Y-%m-%d %H:%M:%S"   : pl.Datetime,
            "%Y-%m-%dT%H:%M:%S%.f": pl.Datetime,
            "%Y-%m-%dT%H:%M:%S"   : pl.Datetime,
            "%Y-%m-%d %H:%M"      : pl.Datetime,
            "%Y-%m-%d"            : pl.Date,
            "%d/%m/%Y"            : pl.Date,
            "%Y-%m"               : pl.Date,
            "%Y"                  : pl.Date,
            "%H:%M:%S"            : pl.Time,
            "%H:%M"               : pl.Time,
            "%H"                  : pl.Time
        }
        for fmt, dtype in datetime_formats_to_dtype.items():
            try:
                # Attempt to parse the string column as datetime using the current format
                if df.head(1).select(pl.col(col).str.strptime(dtype, format=fmt, strict=False)).item() is None:
                    continue
                else:
                    self.datetime_formats[col] = fmt
                
                if fmt in ["%H:%M:%S", "%H:%M", "%H"]:
                    df = df.with_columns(
                        (pl.lit("1970-01-01") + " " + pl.col(col))
                        .alias(col)
                    )
                    fmt = "%Y-%m-%d " + fmt
                df = df.with_columns(pl.col(col).str.strptime(pl.Datetime, format=fmt, strict=False).cast(pl.Int64)/1e6) # Time is in microseconds so the timestamp integer is divided by 1e6
                return df
            except:
                continue
        return df

    def _infer_and_convert_time_columns(self, df):
        for col in df.columns:
            col_dtype = df[col].dtype

            if col_dtype in [pl.Date, pl.Datetime, pl.Time]:
                self.datetime_features = self.datetime_features + tuple([col]) if col not in self.datetime_features else self.datetime_features
            elif col_dtype == pl.Utf8:
                # Check if the column contains date strings
                sample_values = df[col].head(10).to_list()  # Sample a few values for inference
                if all(self._is_date_string(value) for value in sample_values if value is not None):
                    self.datetime_features = self.datetime_features + tuple([col]) if col not in self.datetime_features else self.datetime_features

        # Convert inferred time columns to their respective time data types
        for col in self.datetime_features:
            if df[col].dtype == pl.Utf8:
                # Attempt to parse the string column as datetime using the list of formats
                df = self._try_convert_to_datetime(df, col)
            elif df[col].dtype == pl.Date:
                # Convert Date to Datetime if needed
                df = df.with_columns(pl.col(col).cast(pl.Datetime))

        return df

    def fit(self, data):
        self.dividers = []
        data = self._infer_and_convert_time_columns(data)

        if self.scaling in ["normalize", "quantile", "kbins"]:
            self.time_parameters = [data.select(self.datetime_features).min(), 
                                    data.select(self.datetime_features).max()]
        elif self.scaling == "standardize":
            self.time_parameters = [data.select(self.datetime_features).mean(), 
                                    data.select(self.datetime_features).std()] 
        return data.lazy()
    
    def transform(self, data):
        if isinstance(data, pl.LazyFrame):
            data = data.collect()
        
        data = data.sort(self.datetime_features[0])
        data = self._infer_and_convert_time_columns(data) # Returns data with time columns converted to integers
        data = data.with_columns(pl.col(self.datetime_features).interpolate()) # Linear interpolation

        if self.scaling in ["normalize", "quantile", "kbins"]:
            for col in self.datetime_features:
                col_min = self.time_parameters[0][col].item()
                col_max = self.time_parameters[1][col].item()
                data = data.with_columns((pl.col(col) - col_min) / (col_max - col_min))
        elif self.scaling == "standardize":
            for col in self.datetime_features:
                col_mean = self.time_parameters[0][col].item()
                col_std  = self.time_parameters[1][col].item()
                data = data.with_columns((pl.col(col) - col_mean) /  col_std) 
        return data

    def inverse_transform(self, data):
        if self.scaling in ["normalize", "quantile", "kbins"]:
            for col in self.datetime_features:
                col_min = self.time_parameters[0][col].item()
                col_max = self.time_parameters[1][col].item()
                data = data.with_columns(pl.col(col) * (col_max - col_min) + col_min)
        elif self.scaling == "standardize":
            for col in self.datetime_features:
                col_mean = self.time_parameters[0][col].item()
                col_std  = self.time_parameters[1][col].item()
                data = data.with_columns(pl.col(col) *  col_std + col_mean) 

        for col, fmt in self.datetime_formats.items():
            data = data.with_columns(pl.from_epoch(pl.col(col)*1e6, time_unit="us")) # Convert to Datetime format
            data = data.with_columns(pl.col(col).dt.strftime(fmt).alias(col)) # Convert to string format and original datetime/date/time format format
        
        return data