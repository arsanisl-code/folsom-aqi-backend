import pandas as pd
import os

def aqi_color(aqi):
    if aqi <= 50:  return "#00e400", "#000000"  # Green, Black font
    if aqi <= 100: return "#ffff00", "#000000"  # Yellow, Black font
    if aqi <= 150: return "#ff7e00", "#000000"  # Orange, Black font
    if aqi <= 200: return "#ff0000", "#ffffff"  # Red, White font
    if aqi <= 300: return "#8f3f97", "#ffffff"  # Purple, White font
    return "#7e0023", "#ffffff"                # Maroon, White font

def generate():
    csv_path = "models/validation_predictions.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Sort for better readability: by horizon then timestamp
    df = df.sort_values(['horizon_h', 'timestamp'])

    html = """
    <html>
    <head>
    <title>Folsom AQI - Validation Report</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f4f9; padding: 20px; }
        h1 { color: #333; }
        .stats { margin-bottom: 20px; color: #666; }
        table { border-collapse: collapse; width: 100%; max-width: 900px; background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #333; color: white; position: sticky; top: 0; }
        .aqi-pill { padding: 4px 10px; border-radius: 12px; font-weight: bold; display: inline-block; min-width: 30px; text-align: center; }
        .horizon-group { background: #eee; font-weight: bold; }
    </style>
    </head>
    <body>
        <h1>Folsom AQI Forecast - Accuracy Report</h1>
        <p class="stats">Showing last 30 days of out-of-sample predictions.</p>
    """

    for horizon in sorted(df['horizon_h'].unique()):
        html += f"<h2>{horizon}-Hour Forecast Integrity</h2>"
        html += "<table>"
        html += "<tr><th>Timestamp</th><th>Actual AQI</th><th>Predicted AQI</th><th>Error</th></tr>"
        
        subset = df[df['horizon_h'] == horizon].tail(100) # Show last 100 per horizon to keep file size sane
        
        for _, row in subset.iterrows():
            act = int(row['actual_aqi'])
            prd = int(row['predicted_aqi'])
            err = abs(act - prd)
            
            act_bg, act_fg = aqi_color(act)
            prd_bg, prd_fg = aqi_color(prd)
            
            html += f"<tr>"
            html += f"<td>{row['timestamp']}</td>"
            html += f"<td><span class='aqi-pill' style='background:{act_bg}; color:{act_fg}'>{act}</span></td>"
            html += f"<td><span class='aqi-pill' style='background:{prd_bg}; color:{prd_fg}'>{prd}</span></td>"
            html += f"<td>{err}</td>"
            html += f"</tr>"
        
        html += "</table><br><br>"

    html += "</body></html>"

    with open("validation_report.html", "w") as f:
        f.write(html)
    print("Report generated: validation_report.html")

if __name__ == "__main__":
    generate()
