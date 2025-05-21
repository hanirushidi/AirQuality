from flask import Flask, render_template, request, session, jsonify, make_response, abort # Ensure session is imported
import os
import pandas as pd
from utils import data_loader
from utils import preprocessor # Add this import
import pdfkit # If you're still using it

# Import Flask-Session
from flask_session import Session

# Imports for Correlation Analysis (and Outlier Detection)
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import numpy as np # For selecting numeric types

# Imports for Clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

app = Flask(__name__)

# --- Flask-Session Configuration ---
# IMPORTANT: Change this secret key for production!
app.secret_key = os.urandom(24) # Or a strong, static secret key

# Configure session type to filesystem (server-side)
app.config['SESSION_TYPE'] = 'filesystem'
# Define the directory to store session files
# Ensure this directory exists and is writable by your application
app.config['SESSION_FILE_DIR'] = os.path.join(os.path.dirname(__file__), 'flask_session')
app.config['SESSION_PERMANENT'] = False # Session expires when browser closes, or set a lifetime
app.config['SESSION_USE_SIGNER'] = True  # Encrypts the session cookie
app.config['SESSION_KEY_PREFIX'] = 'airquality_session:' # Optional: prefix for session keys

# Create the session directory if it doesn't exist
if not os.path.exists(app.config['SESSION_FILE_DIR']):
    os.makedirs(app.config['SESSION_FILE_DIR'])

# Initialize the Flask-Session extension
sess = Session()
sess.init_app(app)
# --- End Flask-Session Configuration ---

# Define DATA_DIR and DEFAULT_CSV_PATH (ensure these are correctly defined)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, 'AirQualityUCI.csv')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_data', methods=['GET', 'POST'])
def load_data_route():
    if request.method == 'POST':
        # For now, we'll just use the default dataset
        # File upload logic can be added later if needed
        file_path = DEFAULT_CSV_PATH
        if not os.path.exists(file_path):
            return render_template('load_data.html', error="Default dataset AirQualityUCI.csv not found in data/ directory.")

        df_preview, error = data_loader.load_and_preview_data(file_path)
        if error:
            return render_template('load_data.html', error=error)
        
        # Store dataframe in session (for simplicity, consider alternatives for large data)
        # For large datasets, it's better to store file path and reload, or use a database.
        session['df_loaded'] = True # Flag to indicate data is loaded
        # Pandas DataFrame is not directly JSON serializable for session if too large.
        # Storing a sample or just confirmation. Full processing will happen in next steps.
        return render_template('load_data.html', table=df_preview.to_html(classes='table table-striped table-hover', max_rows=10), file_loaded=True)

    return render_template('load_data.html', file_loaded=session.get('df_loaded', False))

@app.route('/preprocessing')
def preprocessing_route():
    if not session.get('df_loaded'):
        return render_template('preprocessing.html', error="Please load data first from the 'Load Dataset' page.")

    # Load the full dataset
    raw_df, error = data_loader.get_full_data(DEFAULT_CSV_PATH)
    if error:
        return render_template('preprocessing.html', error=f"Error loading data for preprocessing: {error}")
    if raw_df is None:
        return render_template('preprocessing.html', error="Failed to load data for preprocessing.")

    # Perform preprocessing
    try:
        processed_hourly_df, processed_daily_df = preprocessor.preprocess_data(raw_df.copy())
        daily_data_json = processed_daily_df.reset_index().to_json(orient='split', date_format='iso')
        
        # Update session with atomic operation
        session.update({
            'processed_daily_data_json': daily_data_json,
            'data_preprocessed': True
        })
        session.permanent = True  # Add this line
        session.modified = True
        
        # Verify session storage
        if not session.get('data_preprocessed'):
            raise RuntimeError("Session storage failed")
        
        # Clear any cached previous errors
        session.pop('_flashes', None)
        
        # Prepare previews
        hourly_preview_html = processed_hourly_df.head(10).to_html(classes='table table-striped table-hover table-sm', max_rows=10)
        daily_preview_html = processed_daily_df.head(10).to_html(classes='table table-striped table-hover table-sm', max_rows=10)

        return render_template('preprocessing.html',
                           message="Data preprocessing complete!",
                           hourly_preview=hourly_preview_html,
                           daily_preview=daily_preview_html)
    except Exception as e:
        session['data_preprocessed'] = False  # Ensure flag is cleared on error
        return render_template('preprocessing.html', error=f"An error occurred during preprocessing: {str(e)}")

@app.route('/trend_analysis')
def trend_analysis_route():
    if not session.get('df_loaded'):
        return render_template('trend_analysis.html', error="Please load data first.")
    
    # Enhanced verification with error logging
    try:
        if not session.get('data_preprocessed') or 'processed_daily_data_json' not in session:
            raise ValueError("Preprocessing not completed")
            
        df = pd.read_json(session['processed_daily_data_json'], orient='split')
        preview_html = df.head(10).to_html(classes='table table-striped table-hover table-sm', max_rows=10)
        return render_template('trend_analysis.html', 
                            message="Daily Aggregated Data (First 10 Rows)",
                            data_preview=preview_html)
    except Exception as e:
        print(f"Session Debug - Trend Analysis: {dict(session)}")  # Add debug logging
        session.clear()
        return render_template('trend_analysis.html', 
                            error=f"System error: {str(e)}. Please restart the workflow.")

@app.route('/outlier_detection')
def outlier_detection_route():
    if not session.get('df_loaded'):
        return render_template('outlier_detection.html', error="Please load data first.")
    if not session.get('data_preprocessed') or 'processed_daily_data_json' not in session:
        return render_template('outlier_detection.html', error="Please preprocess data first.")

    try:
        df_json = session['processed_daily_data_json']
        df = pd.read_json(df_json, orient='split')

        if 'DateTime' in df.columns:
            try:
                df['DateTime'] = pd.to_datetime(df['DateTime'])
                df.set_index('DateTime', inplace=True)
            except Exception as e:
                print(f"Warning: Could not set DateTime index for outlier detection: {e}")
        
        numeric_df = df.select_dtypes(include=np.number)

        if numeric_df.empty:
            return render_template('outlier_detection.html', error="No numeric data available for outlier detection.")

        outliers_summary = {}
        # plots_data = [] # This variable is not used

        num_cols = len(numeric_df.columns)
        if num_cols == 0:
             return render_template('outlier_detection.html', error="No numeric columns found for outlier detection.")

        n_plot_cols = min(3, num_cols) 
        n_plot_rows = (num_cols + n_plot_cols - 1) // n_plot_cols

        fig, fig_axes = plt.subplots(n_plot_rows, n_plot_cols, figsize=(5 * n_plot_cols, 4 * n_plot_rows))
        
        # Ensure fig_axes is a flat list/array of Axes objects
        flat_axes_list = []
        if isinstance(fig_axes, np.ndarray):
            flat_axes_list = fig_axes.flatten()
        else: # Single AxesSubplot object
            flat_axes_list = [fig_axes]

        for i, col_name in enumerate(numeric_df.columns): # Renamed col to col_name for clarity
            if i >= len(flat_axes_list): 
                # This should not happen with the current n_plot_rows/cols logic
                # but is a safeguard.
                break 
            
            ax = flat_axes_list[i] # Get the current axis

            Q1 = numeric_df[col_name].quantile(0.25)
            Q3 = numeric_df[col_name].quantile(0.75)
            IQR_val = Q3 - Q1 # Renamed IQR to IQR_val to avoid conflict if there's a column named IQR
            lower_bound = Q1 - 1.5 * IQR_val
            upper_bound = Q3 + 1.5 * IQR_val
            
            column_outliers = numeric_df[(numeric_df[col_name] < lower_bound) | (numeric_df[col_name] > upper_bound)][col_name]
            outliers_summary[col_name] = {
                'count': len(column_outliers),
                'values': column_outliers.to_list() 
            }
            
            sns.boxplot(y=numeric_df[col_name], ax=ax, whis=1.5)
            ax.set_title(f'Box Plot for {col_name}', fontsize=10)
            ax.set_ylabel(col_name)

        # Hide any unused subplots
        # Ensure 'i' is defined if the loop ran. num_cols > 0 check ensures this.
        if num_cols > 0: 
            for j_idx in range(i + 1, len(flat_axes_list)): # Renamed j to j_idx
                fig.delaxes(flat_axes_list[j_idx])

        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close(fig) 

        return render_template('outlier_detection.html',
                               plot_url=plot_url,
                               outliers_summary=outliers_summary)

    except Exception as e:
        # For better debugging, you might want to print the full traceback
        import traceback
        print(f"Error in outlier_detection_route: {str(e)}")
        traceback.print_exc() # This will print the full traceback to your Flask console
        return render_template('outlier_detection.html', error=f"An error occurred during outlier detection: {str(e)}. You might need to restart the workflow.")

@app.route('/correlation_analysis')
def correlation_analysis_route():
    if not session.get('df_loaded'):
        return render_template('correlation_analysis.html', error="Please load data first.")
    if not session.get('data_preprocessed') or 'processed_daily_data_json' not in session:
        return render_template('correlation_analysis.html', error="Please preprocess data first.")

    try:
        df_json = session['processed_daily_data_json']
        df = pd.read_json(df_json, orient='split')

        # Ensure DateTime index is handled if it became a column after to_json/read_json
        # If 'DateTime' is a column and was the original index:
        if 'DateTime' in df.columns:
            try:
                df['DateTime'] = pd.to_datetime(df['DateTime'])
                df.set_index('DateTime', inplace=True)
            except Exception as e:
                # If DateTime conversion or setting index fails, proceed with numeric columns only
                print(f"Warning: Could not set DateTime index for correlation: {e}")
        
        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=np.number)

        if numeric_df.empty or numeric_df.shape[1] < 2:
            return render_template('correlation_analysis.html', error="Not enough numeric data available for correlation analysis.")

        # Calculate Pearson correlation matrix
        corr_matrix = numeric_df.corr(method='pearson')
        corr_matrix_html = corr_matrix.to_html(classes='table table-striped table-hover table-sm', float_format='%.2f', border=0)

        # Generate heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Pearson Correlation Heatmap of Daily Aggregated Data', fontsize=15)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout() # Adjust layout to prevent labels from being cut off

        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close() # Close the plot to free memory

        return render_template('correlation_analysis.html',
                               plot_url=plot_url,
                               corr_matrix_html=corr_matrix_html)

    except Exception as e:
        # Log the exception for debugging
        print(f"Error in correlation_analysis_route: {e}")
        # Potentially clear parts of session if data is corrupt, or just show error
        # session.pop('processed_daily_data_json', None) # Example: clear specific item
        # session.pop('data_preprocessed', None)
        return render_template('correlation_analysis.html', error=f"An error occurred during correlation analysis: {str(e)}. You might need to restart the workflow.")

@app.route('/clustering')
def clustering_route():
    if not session.get('df_loaded'):
        return render_template('clustering.html', error="Please load data first.")
    if not session.get('data_preprocessed') or 'processed_daily_data_json' not in session:
        return render_template('clustering.html', error="Please preprocess data first.")

    try:
        df_json = session['processed_daily_data_json']
        df = pd.read_json(df_json, orient='split')

        if 'DateTime' in df.columns:
            try:
                df['DateTime'] = pd.to_datetime(df['DateTime'])
                df.set_index('DateTime', inplace=True)
            except Exception as e:
                print(f"Warning: Could not set DateTime index for clustering: {e}")
        
        numeric_df = df.select_dtypes(include=np.number).copy() # Use .copy() to avoid SettingWithCopyWarning

        if numeric_df.empty or numeric_df.shape[0] < 2 or numeric_df.shape[1] < 1: # Need at least 1 feature and 2 samples
            return render_template('clustering.html', error="Not enough numeric data or samples available for clustering.")

        # Fill NaN values that might remain (e.g., if a column was all NaN before daily aggregation)
        numeric_df.fillna(numeric_df.mean(), inplace=True) # Fill with mean
        numeric_df.dropna(axis=1, how='all', inplace=True) # Drop columns that are still all NaN
        numeric_df.dropna(axis=0, how='any', inplace=True) # Drop rows with any NaN left in selected numeric cols

        if numeric_df.empty or numeric_df.shape[0] < 2 or numeric_df.shape[1] < 1:
             return render_template('clustering.html', error="Data became empty after NaN handling. Not enough data for clustering.")


        # --- K-Means Clustering ---
        # 1. Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        # 2. Determine number of clusters (e.g., 3 for this example)
        #    In a real scenario, you might use the Elbow method or Silhouette score to find optimal k
        n_clusters = min(3, len(numeric_df) -1) # Ensure n_clusters is less than n_samples
        if n_clusters < 2 : # K-Means needs at least 2 clusters, and more practically, data to support it
            return render_template('clustering.html', error=f"Not enough distinct data points to form {n_clusters} clusters. Need at least {n_clusters+1} data points.")


        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(scaled_data)
        numeric_df['Cluster'] = cluster_labels

        # --- Visualization (using PCA for 2D plot) ---
        plot_url = None
        if scaled_data.shape[1] >= 2: # PCA needs at least 2 features
            pca = PCA(n_components=2, random_state=42)
            principal_components = pca.fit_transform(scaled_data)
            pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
            pca_df['Cluster'] = cluster_labels

            plt.figure(figsize=(10, 7))
            sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=100, alpha=0.7)
            plt.title(f'K-Means Clustering (k={n_clusters}) with PCA Visualization', fontsize=15)
            plt.xlabel('Principal Component 1 (PC1)', fontsize=12)
            plt.ylabel('Principal Component 2 (PC2)', fontsize=12)
            plt.grid(True)
            plt.tight_layout()

            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')
            plt.close()
        else: # Handle case with only 1 feature (can't do 2D PCA plot)
            # For 1 feature, a histogram or density plot per cluster might be better
            # For simplicity, we'll just note that PCA plot isn't possible
            print("PCA plot not generated: Less than 2 features available after preprocessing for clustering.")


        # Cluster summary
        cluster_summary = numeric_df['Cluster'].value_counts().sort_index().to_dict()
        cluster_summary_html = pd.DataFrame.from_dict(cluster_summary, orient='index', columns=['Count']) \
                                .reset_index() \
                                .rename(columns={'index': 'Cluster'}) \
                                .to_html(classes='table table-striped table-hover table-sm', index=False)


        return render_template('clustering.html',
                               plot_url=plot_url,
                               cluster_summary_html=cluster_summary_html,
                               num_clusters_used=n_clusters)

    except Exception as e:
        import traceback
        print(f"Error in clustering_route: {e}")
        traceback.print_exc()
        return render_template('clustering.html', error=f"An error occurred during clustering: {str(e)}. You might need to restart the workflow.")

@app.route('/report')
def report_route():
    if not session.get('df_loaded'):
        return render_template('report.html', error="Please load data first.")
    # Add report generation logic here
    return render_template('report.html', message="Downloadable Report section: Implement PDF/HTML report generation.")

@app.route('/contact')
def contact_route():
    return render_template('contact.html')

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        data = request.get_json()
        report_format = data.get('format', 'pdf')
        
        # Get processed data from session
        if 'processed_daily_data_json' not in session:
            return jsonify({'success': False, 'error': 'No processed data available'})
            
        df = pd.read_json(session['processed_daily_data_json'], orient='split')
        
        # Generate report based on format
        if report_format == 'pdf':
            # Implement PDF generation logic
            return jsonify({
                'success': True,
                'download_url': '/download_report.pdf'
            })
        else:
            # Render HTML directly instead of triggering download
            rendered = render_template('report_template.html',
                                     title="Air Quality Report",
                                     data=df.head(20).to_dict('records'))
            return jsonify({
                'success': True,
                'html_content': rendered
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download_report.pdf')
def download_pdf():
    try:
        if 'processed_daily_data_json' not in session:
            abort(404)
            
        df = pd.read_json(session['processed_daily_data_json'], orient='split')
        rendered = render_template('report_template.html', 
                                 title="Air Quality Report",
                                 data=df.head(20).to_dict('records'))
        
        pdf = pdfkit.from_string(rendered, False)
        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'  # Fixed missing quotes
        response.headers['Content-Disposition'] = 'attachment; filename=air_quality_report.pdf'
        return response
    except Exception as e:
        return str(e), 500

@app.route('/download_report.html')
def download_html():
    try:
        if 'processed_daily_data_json' not in session:
            abort(404)
            
        df = pd.read_json(session['processed_daily_data_json'], orient='split')
        rendered = render_template('report_template.html',
                                 title="Air Quality Report",
                                 data=df.head(20).to_dict('records'))
        
        response = make_response(rendered)
        response.headers['Content-Type'] = 'text/html'
        response.headers['Content-Disposition'] = 'attachment; filename=air_quality_report.html'
        return response
    except Exception as e:
        return str(e), 500
if __name__ == '__main__':
    app.run(debug=True, port=5001)
