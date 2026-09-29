from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import polars as pl
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import sys

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import functions from existing pipeline
from Production import (
    CACHE_DIR,
    DATA_DIR,
    Features_Engineering,
    Load_Clean_CreateMaster,
    get_Sales,
    model,
    testing_date,
    transformer,
)
from datetime import date

app = Flask(__name__, static_folder='static', static_url_path='')
# Enable CORS for all routes (important for React on port 5173)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Global variables for cached data in memory
_predictions = None
_master_data = None
_items_list = None

def get_cached_predictions():
    global _predictions
    if _predictions is None:
        print("Calculating / loading predictions (this may take a few seconds on first run)...")
        _predictions = get_Sales()
    return _predictions

def get_cached_master():
    global _master_data
    if _master_data is None:
        print("Loading / creating master data...")
        _master_data = Load_Clean_CreateMaster()
    return _master_data

def get_cached_items():
    global _items_list
    if _items_list is None:
        master = get_cached_master()
        # Extract unique items with their metadata
        items_df = master.select(["item_nbr", "family", "class", "perishable"]).unique(subset=["item_nbr"]).sort("item_nbr")
        _items_list = [
            {
                "item_nbr": int(row["item_nbr"]),
                "family": str(row["family"]),
                "class": int(row["class"]),
                "perishable": int(row["perishable"])
            }
            for row in items_df.iter_rows(named=True)
        ]
    return _items_list

@app.route('/api/kpi-summary', methods=['GET'])
def get_kpi_summary():
    try:
        preds = get_cached_predictions()
        master = get_cached_master()
        
        # Calculate predicted total sales
        day_cols = [f"Day_{i}" for i in range(1, 29)]
        # Total forecasted sales
        total_forecasted = float(preds.select(day_cols).sum().sum_horizontal()[0])
        avg_daily_forecast = total_forecasted / 28.0
        
        # Unique items and families
        unique_items = len(get_cached_items())
        unique_families = master.get_column("family").n_unique()
        
        # Historical stats (last 30 days before testing_date)
        hist_30d = master.filter(
            (pl.col("date") <= testing_date) & 
            (pl.col("date") > testing_date - timedelta(days=30))
        )
        total_hist_sales_30d = float(hist_30d.select("unit_sales").sum()[0, 0])
        
        # Top 5 families by predicted sales
        family_preds = preds.group_by("family").agg(
            [pl.col(col).sum() for col in day_cols]
        )
        family_preds = family_preds.with_columns(
            pl.sum_horizontal(day_cols).alias("total_pred")
        ).sort("total_pred", descending=True)
        
        top_families = [
            {"family": str(row["family"]), "sales": float(row["total_pred"])}
            for row in family_preds.head(5).iter_rows(named=True)
        ]
        
        # Current oil price (on testing_date or last available before it)
        oil_df = master.filter(pl.col("date") <= testing_date).sort("date", descending=True)
        latest_oil_price = None
        if len(oil_df) > 0:
            latest_oil_price = oil_df.get_column("oil_price")[0]
            if latest_oil_price is not None:
                latest_oil_price = float(latest_oil_price)
                
        # Latest transaction count
        trans_df = master.filter((pl.col("date") == testing_date) & (pl.col("transactions") > 0))
        latest_transactions = None
        if len(trans_df) > 0:
            latest_transactions = int(trans_df.get_column("transactions")[0])
            
        return jsonify({
            "success": True,
            "data": {
                "totalForecastedSales": total_forecasted,
                "avgDailyForecast": avg_daily_forecast,
                "uniqueItems": unique_items,
                "uniqueFamilies": unique_families,
                "totalHistoricalSales30d": total_hist_sales_30d,
                "topFamilies": top_families,
                "latestOilPrice": latest_oil_price,
                "latestTransactions": latest_transactions,
                "testingDate": testing_date.strftime("%Y-%m-%d")
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/predictions/summary', methods=['GET'])
def get_predictions_summary():
    try:
        preds = get_cached_predictions()
        day_cols = [f"Day_{i}" for i in range(1, 29)]
        
        # Aggregate total predictions for each of the 28 days
        daily_sums = preds.select(day_cols).sum()
        
        dates = [(testing_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 29)]
        daily_data = []
        for i, col in enumerate(day_cols):
            daily_data.append({
                "day": i + 1,
                "date": dates[i],
                "sales": float(daily_sums.get_column(col)[0])
            })
            
        # Group by family and aggregate
        family_preds = preds.group_by("family").agg(
            [pl.col(col).sum() for col in day_cols]
        ).sort("family")
        
        family_data = {}
        for row in family_preds.iter_rows(named=True):
            f_name = str(row["family"])
            f_sales = [float(row[col]) for col in day_cols]
            family_data[f_name] = {
                "total": sum(f_sales),
                "daily": f_sales
            }
            
        return jsonify({
            "success": True,
            "data": {
                "dailyForecast": daily_data,
                "familyForecast": family_data,
                "dates": dates
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    try:
        family = request.args.get('family')
        item_nbr = request.args.get('item_nbr')
        perishable = request.args.get('perishable')
        
        preds = get_cached_predictions()
        
        # Apply filters
        if family:
            preds = preds.filter(pl.col("family") == family)
        if item_nbr:
            try:
                preds = preds.filter(pl.col("item_nbr") == int(item_nbr))
            except ValueError:
                pass
        if perishable not in (None, ''):
            try:
                preds = preds.filter(pl.col("perishable") == int(perishable))
            except ValueError:
                pass
                
        # Limit rows to avoid huge JSON transfer (e.g. max 500 rows unless specified otherwise)
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        
        total_rows = len(preds)
        preds_subset = preds.slice(offset, limit)
        
        day_cols = [f"Day_{i}" for i in range(1, 29)]
        
        results = []
        for row in preds_subset.iter_rows(named=True):
            predictions_list = [float(row[col]) for col in day_cols]
            results.append({
                "item_nbr": int(row["item_nbr"]),
                "family": str(row["family"]),
                "class": int(row["class"]),
                "perishable": int(row["perishable"]),
                "predictions": predictions_list,
                "total": sum(predictions_list)
            })
            
        return jsonify({
            "success": True,
            "data": results,
            "pagination": {
                "total": total_rows,
                "limit": limit,
                "offset": offset
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/sales-history', methods=['GET'])
def get_sales_history():
    try:
        family = request.args.get('family')
        item_nbr = request.args.get('item_nbr')
        days = int(request.args.get('days', 60))  # Default to last 60 days
        
        master = get_cached_master()
        
        # Filter for historical dates (<= testing_date)
        min_hist_date = testing_date - timedelta(days=days)
        hist = master.filter((pl.col("date") >= min_hist_date) & (pl.col("date") <= testing_date))
        
        if family:
            hist = hist.filter(pl.col("family") == family)
        if item_nbr:
            try:
                hist = hist.filter(pl.col("item_nbr") == int(item_nbr))
            except ValueError:
                pass
                
        # Group by date to get aggregate time series
        daily_hist = hist.group_by("date").agg(
            pl.col("unit_sales").sum().alias("sales")
        ).sort("date")
        
        results = [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "sales": float(row["sales"])
            }
            for row in daily_hist.iter_rows(named=True)
        ]
        
        return jsonify({
            "success": True,
            "data": results
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/market-data', methods=['GET'])
def get_market_data():
    try:
        days = int(request.args.get('days', 90))
        master = get_cached_master()
        
        min_date = testing_date - timedelta(days=days)
        
        # Sub-select unique dates with oil_price and transactions
        # Note: oil price and transactions are store/day-level, not item-level.
        # So we group by date.
        market_df = master.filter(
            (pl.col("date") >= min_date) & (pl.col("date") <= testing_date)
        ).group_by("date").agg([
            pl.col("oil_price").first().alias("oil_price"),
            pl.col("transactions").first().alias("transactions")
        ]).sort("date")
        
        results = []
        for row in market_df.iter_rows(named=True):
            oil = row["oil_price"]
            trans = row["transactions"]
            results.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "oil_price": float(oil) if oil is not None else None,
                "transactions": int(trans) if trans is not None else 0
            })
            
        return jsonify({
            "success": True,
            "data": results
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/promotions', methods=['GET'])
def get_promotions_analytics():
    try:
        master = get_cached_master()
        preds = get_cached_predictions()
        
        # 1. Historical promotion distribution by family
        hist_promo = master.filter(pl.col("date") <= testing_date)
        total_by_family = hist_promo.group_by("family").agg([
            pl.col("onpromotion").sum().alias("promo_count"),
            pl.count().alias("total_count")
        ])
        
        family_promo_stats = []
        for row in total_by_family.iter_rows(named=True):
            f_name = row["family"]
            promo_cnt = row["promo_count"]
            total_cnt = row["total_count"]
            family_promo_stats.append({
                "family": str(f_name),
                "promo_ratio": float(promo_cnt / total_cnt) if total_cnt > 0 else 0,
                "promo_days_count": int(promo_cnt)
            })
            
        # Sort by promo ratio descending
        family_promo_stats.sort(key=lambda x: x["promo_ratio"], reverse=True)
        
        # 2. Predicted lift analysis (Comparing items on promo vs not on promo in forecast)
        # We can analyze this using the predictions DataFrame or historical master
        # Let's check average sales when on promotion vs off promotion in historical data
        promo_impact = hist_promo.group_by("onpromotion").agg(
            pl.col("unit_sales").mean().alias("avg_sales")
        )
        
        promo_lift = {}
        for row in promo_impact.iter_rows(named=True):
            is_promo = bool(row["onpromotion"])
            avg_sales = float(row["avg_sales"])
            promo_lift["promo" if is_promo else "no_promo"] = avg_sales
            
        # Calculate percentage lift
        lift_percentage = 0.0
        if "no_promo" in promo_lift and promo_lift["no_promo"] > 0:
            lift_percentage = ((promo_lift.get("promo", 0.0) - promo_lift["no_promo"]) / promo_lift["no_promo"]) * 100.0
            
        # 3. Upcoming promotions in next 28 days
        # We can read this from PromotionCalendar.csv directly for dates > testing_date
        promotions_path = DATA_DIR / "PromotionCalendar.csv"
        upcoming_promos = pl.read_csv(promotions_path, try_parse_dates=True).filter(
            (pl.col("date") > testing_date) & 
            (pl.col("date") <= testing_date + timedelta(days=28)) &
            (pl.col("onpromotion") == True)
        )
        
        # Group by date to see promo intensity
        upcoming_daily = upcoming_promos.group_by("date").agg(
            pl.count().alias("promo_items_count")
        ).sort("date")
        
        upcoming_promo_timeline = [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "promo_count": int(row["promo_items_count"])
            }
            for row in upcoming_daily.iter_rows(named=True)
        ]
        
        return jsonify({
            "success": True,
            "data": {
                "familyPromoStats": family_promo_stats[:10], # top 10
                "promoLift": {
                    "avgSalesPromo": promo_lift.get("promo", 0.0),
                    "avgSalesNoPromo": promo_lift.get("no_promo", 0.0),
                    "liftPercentage": lift_percentage
                },
                "upcomingPromoTimeline": upcoming_promo_timeline
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/holidays', methods=['GET'])
def get_holidays_events():
    try:
        holidays_path = DATA_DIR / "holidays.csv"
        holidays_df = pl.read_csv(holidays_path, try_parse_dates=True)
        
        # Filter for holidays around our timeline (historical 90 days to forecast 28 days)
        min_date = testing_date - timedelta(days=90)
        max_date = testing_date + timedelta(days=28)
        
        relevant_holidays = holidays_df.filter(
            (pl.col("date") >= min_date) & 
            (pl.col("date") <= max_date) & 
            (pl.col("is_Holiday") == 1)
        ).sort("date")
        
        holiday_list = []
        for row in relevant_holidays.iter_rows(named=True):
            h_date = row["date"]
            holiday_list.append({
                "date": h_date.strftime("%Y-%m-%d"),
                "type": "National" if row["is_National"] == 1 else "Local",
                "is_likely_closed": bool(row["is_likely_closed"]),
                "is_past": h_date <= testing_date,
                "days_offset": (h_date - testing_date).days
            })
            
        return jsonify({
            "success": True,
            "data": holiday_list
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/items', methods=['GET'])
def get_items_catalog():
    try:
        return jsonify({
            "success": True,
            "data": get_cached_items()
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/predictions/details', methods=['GET'])
def get_item_details():
    try:
        item_nbr = int(request.args.get('item_nbr'))
        
        # Load predictions
        preds = get_cached_predictions()
        item_pred = preds.filter(pl.col("item_nbr") == item_nbr)
        
        if len(item_pred) == 0:
            return jsonify({"success": False, "error": "Item not found"}), 404
            
        # Get promotions and history for this item
        master = get_cached_master()
        
        # Promotions for next 28 days
        promo_df = master.filter(
            (pl.col("item_nbr") == item_nbr) & 
            (pl.col("date") > testing_date) & 
            (pl.col("date") <= testing_date + timedelta(days=28))
        ).sort("date")
        
        promotions_list = [bool(row["onpromotion"]) for row in promo_df.iter_rows(named=True)]
        
        while len(promotions_list) < 28:
            promotions_list.append(False)
            
        row = item_pred.row(0, named=True)
        day_cols = [f"Day_{i}" for i in range(1, 29)]
        predictions_list = [float(row[col]) for col in day_cols]
        
        # Get history
        hist_df = master.filter(
            (pl.col("item_nbr") == item_nbr) & 
            (pl.col("date") <= testing_date) & 
            (pl.col("date") > testing_date - timedelta(days=30))
        ).sort("date")
        
        history_list = [
            {"date": r["date"].strftime("%Y-%m-%d"), "sales": float(r["unit_sales"])}
            for r in hist_df.iter_rows(named=True)
        ]
        
        # Logistics
        sales_7d = sum(predictions_list[:7])
        sales_14d = sum(predictions_list[:14])
        sales_28d = sum(predictions_list)
        
        is_perishable = bool(row["perishable"])
        safety_stock_factor = 1.2 if is_perishable else 1.5
        suggested_stock = sales_7d * safety_stock_factor
        reorder_point = sales_7d * 0.7
        
        return jsonify({
            "success": True,
            "data": {
                "item_nbr": item_nbr,
                "family": str(row["family"]),
                "class": int(row["class"]),
                "perishable": is_perishable,
                "predictions": predictions_list,
                "promotions": promotions_list,
                "history": history_list,
                "logistics": {
                    "sales_7d": sales_7d,
                    "sales_14d": sales_14d,
                    "sales_28d": sales_28d,
                    "suggested_stock": suggested_stock,
                    "reorder_point": reorder_point
                }
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/predictions/simulate', methods=['POST'])
def simulate_predictions():
    try:
        req_data = request.get_json()
        item_nbr = int(req_data.get('item_nbr'))
        custom_promos = req_data.get('promotions')
        
        if not custom_promos or len(custom_promos) != 28:
            return jsonify({"success": False, "error": "Invalid promotions list, must have 28 items"}), 400
            
        features_path = CACHE_DIR / "Master_Features_Engineered_Cache.csv"
        if not features_path.exists():
            get_cached_predictions()
            
        df = pl.read_csv(features_path, try_parse_dates=True)
        item_df = df.filter(pl.col("item_nbr") == item_nbr).sort("date")
        
        if len(item_df) == 0:
            return jsonify({"success": False, "error": "Item features not found"}), 404
            
        dates = item_df.get_column("date").to_list()
        new_promos = []
        for d in dates:
            d_date = d
            if hasattr(d, 'date'):
                d_date = d.date()
            offset_days = (d_date - testing_date).days
            if offset_days > 0 and offset_days <= 28:
                new_promos.append(bool(custom_promos[offset_days - 1]))
            else:
                row_val = item_df.filter(pl.col("date") == d).get_column("onpromotion")[0]
                new_promos.append(bool(row_val))
                
        item_df = item_df.with_columns(pl.Series("onpromotion", new_promos))
        sim_preds = model.predict_sales(X=item_df, transformer=transformer)
        
        if len(sim_preds) == 0:
            return jsonify({"success": False, "error": "Simulation failed"}), 500
            
        row = sim_preds.row(0, named=True)
        day_cols = [f"Day_{i}" for i in range(1, 29)]
        predictions_list = [float(row[col]) for col in day_cols]
        
        return jsonify({
            "success": True,
            "data": {
                "item_nbr": item_nbr,
                "predictions": predictions_list
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/')
def index_route():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    # Initialize cache on startup
    print("Pre-loading dataset caches...")
    get_cached_master()
    get_cached_predictions()
    print("Caches loaded. Starting API server on port 5000...")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
