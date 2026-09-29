import polars as pl
from sklearn.preprocessing import FunctionTransformer
from lightgbm import LGBMRegressor
from datetime import date, timedelta, datetime
import numpy as np
import joblib
import os
from pathlib import Path
from dotenv import load_dotenv

testing_date = date(2017, 7, 18)

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _project_path_from_env(variable_name: str, default: str) -> Path:
    """Return an absolute project path, allowing deployments to override it."""
    configured_path = Path(os.getenv(variable_name, default))
    return configured_path if configured_path.is_absolute() else PROJECT_ROOT / configured_path


DATA_DIR = _project_path_from_env("DYLIA_DATA_DIR", "Deployment_Data_Test")
CACHE_DIR = _project_path_from_env("DYLIA_CACHE_DIR", "Cache")
MODELS_DIR = _project_path_from_env("DYLIA_MODELS_DIR", "Models")

salesData_Path = DATA_DIR / "SalesData.csv"
transactions_Path = DATA_DIR / "transactions.csv"
oil_Path = DATA_DIR / "oil.csv"
items_Path = DATA_DIR / "items.csv"
holidays_Path = DATA_DIR / "holidays.csv"
PromotionCalendar_Path = DATA_DIR / "PromotionCalendar.csv"

Cache_Folder_Path = CACHE_DIR

Cleaning_Date_Path = Cache_Folder_Path / "Cleaning_Date.txt"
Master_Cache_File_Path = Cache_Folder_Path / "Master_Cache.csv"

def Load_Clean_CreateMaster():
    if os.path.isfile(Cleaning_Date_Path) and os.path.isfile(Master_Cache_File_Path):
        with open(Cleaning_Date_Path) as file:
            the_date = datetime.strptime(file.readline().strip(), '%Y-%m-%d').date()
        if the_date == testing_date:
            Master_Cache = pl.read_csv(Master_Cache_File_Path, try_parse_dates=True)
            return Master_Cache
        
    Sales = pl.read_csv(salesData_Path, try_parse_dates=True)
    transactions = pl.read_csv(transactions_Path, try_parse_dates=True)
    oil = pl.read_csv(oil_Path, try_parse_dates=True)
    items = pl.read_csv(items_Path)
    holidays = pl.read_csv(holidays_Path, try_parse_dates=True)
    promotions = pl.read_csv(PromotionCalendar_Path, try_parse_dates=True)

    min_date = testing_date - timedelta(days=139)
    max_date = testing_date + timedelta(days=28)
    items_nbr = items.get_column('item_nbr').unique().sort()

    if Sales.get_column('date').min() > min_date:
        raise ValueError('the Sales min Date should be before '+min_date.strftime('%Y-%m-%d'))
    
    Sales = Sales.filter((pl.col('date') >= min_date) & (pl.col('date') <= testing_date))
    transactions = transactions.filter((pl.col('date') >= min_date) & (pl.col('date') <= testing_date))
    oil = oil.filter((pl.col('date') >= min_date) & (pl.col('date') <= testing_date))
    holidays = holidays.filter((pl.col('date') >= min_date) & (pl.col('date') <= max_date))
    promotions = promotions.filter((pl.col('date') >= min_date) & (pl.col('date') <= max_date))

    delta = timedelta(days=1)

    ####################################
    rows = []
    for item in items_nbr:
        start_date = min_date.replace()
        while start_date <= max_date:
            rows.append(
                {
                    "date" : start_date,
                    "item_nbr" : item,
                    "unit_sales" : 0.0,
                }
            )
            start_date += delta

    zero_sales = pl.DataFrame(rows)

    Sales = pl.concat([zero_sales, Sales]).unique(subset=["date", "item_nbr"], keep="last")
    Sales = Sales.sort(by=["item_nbr", "date"])

    ####################################
    rows = []
    for item in items_nbr:
        start_date = min_date.replace()
        while start_date <= max_date:
            rows.append(
                {
                    "date" : start_date,
                    "item_nbr" : item,
                    "onpromotion" : False,
                }
            )
            start_date += delta

    false_promotions = pl.DataFrame(rows)

    promotions = pl.concat([false_promotions, promotions]).unique(subset=["date", "item_nbr"], keep="last")
    promotions = promotions.sort(by=["item_nbr", "date"])

    #####################################
    rows = []
    start_date = min_date.replace()
    while start_date <= max_date:
        rows.append({
            "date" : start_date,
            "transactions" : 0
        })
        start_date += delta
    zero_transactions = pl.DataFrame(rows)

    transactions = pl.concat([zero_transactions, transactions]).unique(subset=["date"], keep="last")
    transactions = transactions.sort(by="date")

    ####################################
    rows = []
    start_date = min_date.replace()
    while start_date <= max_date:
        rows.append(
            {
                "date" : start_date,
                "oil_price" : None
            }
        )
        start_date += delta
    null_oil = pl.DataFrame(rows)
    null_oil = null_oil.with_columns(pl.col("oil_price").cast(pl.Float64))

    oil = pl.concat([null_oil, oil]).unique(subset=["date"], keep="last")
    oil = oil.sort(by="date")
    oil = oil.with_columns(pl.col("oil_price").forward_fill().backward_fill())

    ####################################
    Master = Sales.join(promotions, on=['date', 'item_nbr'], how='left')
    Master = Master.join(items, on='item_nbr', how='left')
    Master = Master.join(transactions, on='date', how='left')
    Master = Master.join(holidays, on='date', how='left')
    Master = Master.join(oil, on='date', how='left')

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(Cleaning_Date_Path, 'w') as file:
        file.write(testing_date.strftime('%Y-%m-%d'))
    Master.write_csv(Master_Cache_File_Path)

    return Master


Master_Features_Engineered_Cache_Path = CACHE_DIR / "Master_Features_Engineered_Cache.csv"
Features_Engineering_Date_Path = CACHE_DIR / "Features_Engineering_Date.txt"

def Features_Engineering(Master : pl.DataFrame):
    if os.path.isfile(Features_Engineering_Date_Path) and os.path.isfile(Master_Features_Engineered_Cache_Path):
        with open(Features_Engineering_Date_Path) as file:
            the_date = datetime.strptime(file.readline().strip(), '%Y-%m-%d').date()
        if the_date == testing_date:
            Master_Features_Engineered_Cache = pl.read_csv(Master_Features_Engineered_Cache_Path, try_parse_dates=True)
            return Master_Features_Engineered_Cache
            

    Master = Master.with_columns(
        [
            pl.col('unit_sales').log1p(),
            pl.col('transactions').log1p()
        ]
    )

    ######################################
    Master = Master.with_columns([
        pl.col("date").dt.weekday().cast(pl.Int8).alias("day_of_week"),
        (pl.col("date").dt.weekday() >= 5).cast(pl.Int8).alias("is_weekend"),
        (pl.col("date").dt.day() <= 3).cast(pl.Int8).alias("is_month_start"),
        (pl.col("date").dt.day() >= 28).cast(pl.Int8).alias("is_month_end")
    ])

    ######################################
    family_sales = Master.select(["date", "unit_sales", "family"])
    family_sales = family_sales.group_by(["family", "date"]).agg(pl.col("unit_sales").sum()).sort(by=["family", "date"])
    family_sales = family_sales.with_columns(
        (pl.col("unit_sales").cum_sum() / pl.col("unit_sales").cum_count()).over("family").alias("family_mean_sales")
    )
    family_sales = family_sales.drop("unit_sales")

    Master = Master.with_columns([
        (pl.col("unit_sales").cum_sum()/pl.col("unit_sales").cum_count()).over("item_nbr").alias("item_mean_sales"),
        pl.col("unit_sales").cumulative_eval(pl.element().std(ddof=0)).over("item_nbr").alias("item_std_sales")
    ])

    Master = Master.join(family_sales, on=["family", "date"], how="left")

    Master = Master.with_columns(
        (pl.col("item_mean_sales") / pl.col("family_mean_sales").clip(lower_bound=0.01)).alias("sales_vs_family_mean")
    )

    Master = Master.drop("family_mean_sales")

    #########################################
    Master = Master.with_columns([
        pl.col("unit_sales").shift(0).over("item_nbr").alias("lag_1"),
        pl.col("unit_sales").shift(2).over("item_nbr").alias("lag_3"),
        pl.col("unit_sales").shift(4).over("item_nbr").alias("lag_5"),
        pl.col("unit_sales").shift(6).over("item_nbr").alias("lag_7"),
        pl.col("unit_sales").shift(13).over("item_nbr").alias("lag_14"),
        pl.col("unit_sales").shift(29).over("item_nbr").alias("lag_30"),
        pl.col("unit_sales").shift(59).over("item_nbr").alias("lag_60"),
        pl.col("unit_sales").shift(139).over("item_nbr").alias("lag_140")
    ])

    Master = Master.with_columns([
        pl.col("unit_sales").rolling_mean(3).over("item_nbr").alias("rolling_mean_3"),
        pl.col("unit_sales").rolling_mean(7).over("item_nbr").alias("rolling_mean_7"),
        pl.col("unit_sales").rolling_mean(30).over("item_nbr").alias("rolling_mean_30"),
        pl.col("unit_sales").rolling_mean(14).over("item_nbr").alias("rolling_mean_14"),
        pl.col("unit_sales").rolling_mean(140).over("item_nbr").alias("rolling_mean_140"),
        pl.col("unit_sales").rolling_std(7).over("item_nbr").alias("rolling_std_7"),
        pl.col("unit_sales").rolling_std(14).over("item_nbr").alias("rolling_std_14"),
        pl.col("unit_sales").rolling_std(30).over("item_nbr").alias("rolling_std_30"),
        pl.col("unit_sales").rolling_max(7).over("item_nbr").alias("rolling_max_7"),
        pl.col("unit_sales").rolling_min(7).over("item_nbr").alias("rolling_min_7")
    ])

    #########################################
    Master = Master.with_columns([
        pl.col("transactions").shift(0).over("item_nbr").alias("transactions_lag_1"),
        pl.col("transactions").shift(2).over("item_nbr").alias("transactions_lag_3"),
        pl.col("transactions").shift(6).over("item_nbr").alias("transactions_lag_7"),
        pl.col("transactions").shift(13).over("item_nbr").alias("transactions_lag_14"),
        pl.col("transactions").rolling_mean(7).over("item_nbr").alias("transactions_rolling_7"),
        pl.col("transactions").rolling_mean(14).over("item_nbr").alias("transactions_rolling_14"),
        pl.col("transactions").rolling_mean(30).over("item_nbr").alias("transactions_rolling_30")
    ]).with_columns(
        (pl.col("transactions_lag_1")/pl.col("transactions_rolling_30")).clip(lower_bound=0.5).alias("traffic_ratio")
    )

    ##########################################
    Master = Master.with_columns([
        pl.col("oil_price").shift(0).over("item_nbr").alias("oil_price_lag_1"),
        pl.col("oil_price").shift(2).over("item_nbr").alias("oil_price_lag_3"),
        pl.col("oil_price").shift(6).over("item_nbr").alias("oil_price_lag_7"),
        pl.col("oil_price").rolling_mean(7).over("item_nbr").alias("oil_price_rolling_mean_7"),
        pl.col("oil_price").rolling_mean(30).over("item_nbr").alias("oil_price_rolling_mean_30")
    ]).with_columns(
        ((pl.col("oil_price_lag_1") - pl.col("oil_price_lag_7")) / pl.col("oil_price_lag_7")).alias("oil_price_change_7")
    )

    ##########################################
    Master = Master.with_columns([
        pl.col("onpromotion").shift(0).over("item_nbr").alias("promo_lag_1"),
        pl.col("onpromotion").shift(2).over("item_nbr").alias("promo_lag_3"),
        pl.col("onpromotion").shift(6).over("item_nbr").alias("promo_lag_7"),
        pl.col("onpromotion").cast(pl.Int8).rolling_sum(7).over("item_nbr").alias("promo_count_7"),
        pl.col("onpromotion").cast(pl.Int8).rolling_sum(30).over("item_nbr").alias("promo_count_30")
    ])
    Master = Master.with_columns(
        pl.when(pl.col("onpromotion").over("item_nbr") == 1)
        .then(pl.col("date"))
        .otherwise(None)
        .forward_fill()
        .over("item_nbr")
        .alias("last_promo_date")
    ).with_columns(
        (pl.col("date") - pl.col("last_promo_date"))
        .dt.total_days()
        .fill_null(999)
        .alias("days_since_last_promo")
    ).drop("last_promo_date")

    ##########################################
    holidays = pl.read_csv(holidays_Path, try_parse_dates=True)

    holiday_dates = (
        holidays.filter(pl.col("is_Holiday") == 1).select("date").sort("date")["date"].to_list()
    )

    unique_dates = Master["date"].unique().sort().to_list()

    days_since = []
    days_to = []

    for d in unique_dates:
        past = [h for h in holiday_dates if h < d]
        days_since.append((d - past[-1]).days if past else 999)

        future = [h for h in holiday_dates if h > d]
        days_to.append((future[0] - d).days if future else 999)

    holiday_distances = pl.DataFrame({
        "date": unique_dates,
        "days_since_last_holiday": days_since,
        "days_to_next_holiday": days_to,
    })

    Master = Master.join(holiday_distances, on="date", how="left")

    #######################################
    Master = Master.drop_nulls(subset=[
        "rolling_mean_140"
    ])

    with open(Features_Engineering_Date_Path, 'w') as file:
        file.write(testing_date.strftime('%Y-%m-%d'))
    Master.write_csv(Master_Features_Engineered_Cache_Path)

    return Master

######################################################################
Features = [
    'item_nbr',
    'family',
    'class',
    'perishable',
    'has_float_sales',
    'is_likely_closed',
    'day_of_week',
    'is_weekend',
    'is_month_start',
    'is_month_end',
    'item_mean_sales',
    'item_std_sales',
    'sales_vs_family_mean',
    'lag_1',
    'lag_3',
    'lag_5',
    'lag_7',
    'lag_14',
    'lag_30',
    'lag_60',
    'lag_140',
    'rolling_mean_3',
    'rolling_mean_7',
    'rolling_mean_14',
    'rolling_mean_30',
    'rolling_mean_140',
    'rolling_std_7',
    'rolling_std_14',
    'rolling_std_30',
    'rolling_max_7',
    'rolling_min_7',
    'transactions_lag_1',
    'transactions_lag_3',
    'transactions_lag_7',
    'transactions_lag_14',
    'transactions_rolling_7',
    'transactions_rolling_14',
    'transactions_rolling_30',
    'traffic_ratio',
    'oil_price_lag_1',
    'oil_price_lag_3',
    'oil_price_lag_7',
    'oil_price_rolling_mean_7',
    'oil_price_rolling_mean_30',
    'oil_price_change_7',
    'promo_lag_1',
    'promo_lag_3',
    'promo_lag_7',
    'promo_count_7',
    'promo_count_30',
    'days_since_last_promo',
    'days_since_last_holiday',
    'days_to_next_holiday',
    'Future_is_Holiday',
    'Future_is_National',
    'Future_is_Local',
    'Future_is_likely_closed',
    'Future_day_of_week',
    'Future_is_weekend',
    'Future_is_month_start',
    'Future_is_month_end',
    'Future_On_Promotion'
]

Categorical_Features = [
    'item_nbr',
    'family',
    'class'
]

Future_Features_Map = {
    'is_Holiday' : 'Future_is_Holiday',
    'is_National' : 'Future_is_National',
    'is_Local' : 'Future_is_Local',
    'is_likely_closed' : 'Future_is_likely_closed',
    'day_of_week' : 'Future_day_of_week',
    'is_weekend' : 'Future_is_weekend',
    'is_month_start' : 'Future_is_month_start',
    'is_month_end' : 'Future_is_month_end',
    'onpromotion' : 'Future_On_Promotion'
}

transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
horizon = 28

Models_Path = MODELS_DIR

Predictions_Cache_Path = CACHE_DIR / "Predictions_Cache.csv"
Predictions_Date_Path = CACHE_DIR / "Predictions_Date.txt"

class RetailForecastingModel:
    def __init__(self, params, horizon):
        if params != None and len(params) != horizon:
            raise ValueError("the length of params should be the same as the horizon")
        
        self.horizon = horizon
        self.models = []
        for h in range(horizon):
            parameters = (params[h] if params != None else {}) | {
                'objective' : 'regression',
                'metric' : None,
                'device' : 'cpu',
                'n_jobs' : -1,
                'verbose' : -1
            }
            self.models.append(LGBMRegressor(**parameters))
    
    def predict_sales(self, X : pl.DataFrame, transformer : FunctionTransformer):
        sales = []

        for h in range(self.horizon):
            Added_Future_Features_X = X.with_columns(
                [
                    pl.col(oldName).shift(-(h+1)).over('item_nbr').alias(newName) for oldName, newName in Future_Features_Map.items()
                ]
            ).with_columns(
                [
                    pl.col(colName).cast(pl.String).cast(pl.Categorical) for colName in Categorical_Features
                ]
            ).with_columns(
                pl.col('item_nbr').shift(-self.horizon).over('item_nbr').alias('delete')
            )

            Added_Future_Features_X = Added_Future_Features_X.drop_nulls()

            rounding_mask = Added_Future_Features_X.get_column('has_float_sales').to_numpy()

            Added_Future_Features_X_pandas = Added_Future_Features_X.select(Features).to_pandas()

            prediction = self.models[h].predict(Added_Future_Features_X_pandas)

            prediction = transformer.inverse_transform(prediction)

            prediction = np.clip(prediction, a_min=0, a_max=None)

            prediction = np.where(rounding_mask == 0, np.round(prediction), prediction)

            sales.append(prediction)

        sales = np.array(sales).T

        predictions_DataFrame = Added_Future_Features_X.select(['item_nbr', 'family', 'class', 'perishable'])
        predictions_DataFrame = predictions_DataFrame.with_columns(
            [
                    pl.Series(values=sales[:,h]).alias(f'Day_{h+1}') for h in range(self.horizon)
            ]
        )

        return predictions_DataFrame
    
    def load_models(self, models_path : str):
        for h in range(self.horizon):
            self.models[h] = joblib.load(Path(models_path) / f"Model{h+1}.pkl")

model = RetailForecastingModel(params=None, horizon=horizon)
model.load_models(Models_Path)


def get_Sales():
    if os.path.isfile(Predictions_Date_Path) and os.path.isfile(Predictions_Cache_Path):
        with open(Predictions_Date_Path) as file:
            the_date = datetime.strptime(file.readline().strip(), '%Y-%m-%d').date()
        if the_date == testing_date:
            predictions_Cache = pl.read_csv(Predictions_Cache_Path)
            return predictions_Cache
        
    Master = Load_Clean_CreateMaster()
    Master_Features_Engineered = Features_Engineering(Master=Master)

    Predictions = model.predict_sales(X=Master_Features_Engineered, transformer=transformer)

    with open(Predictions_Date_Path, 'w') as file:
        file.write(testing_date.strftime('%Y-%m-%d'))
    Predictions.write_csv(Predictions_Cache_Path)

    return Predictions


Sales_Predictions = get_Sales()
