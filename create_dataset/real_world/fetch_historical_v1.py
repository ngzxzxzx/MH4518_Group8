"""
Historical Market Data Downloader for Barrier Reverse Convertible Pricing
Fetches, processes, and analyzes historical price data for multiple assets
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class HistoricalDataDownloader:
    """
    Download and process historical price data for BRC pricing
    """
    
    def __init__(self, tickers=['NFLX', 'SPOT', 'DIS'], 
                 start_date='2020-01-01',
                 end_date='2025-04-01',
                 output_dir='../../final_dataset/prices/fetch_prices_v1'):
        
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create subdirectories for different data types
        self.raw_dir = f"{output_dir}/raw"
        self.processed_dir = f"{output_dir}/processed"
        self.analysis_dir = f"{output_dir}/analysis"
        self.plots_dir = f"{output_dir}/plots"
        
        for dir_path in [self.raw_dir, self.processed_dir, 
                        self.analysis_dir, self.plots_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
    
    def download_data(self):
        """
        Download historical price data from Yahoo Finance
        """
        print("\n" + "="*60)
        print("DOWNLOADING HISTORICAL PRICE DATA")
        print("="*60)
        
        print(f"Tickers: {self.tickers}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Estimated data points: ~{len(pd.date_range(self.start_date, self.end_date, freq='B'))} trading days")
        
        try:
            # Download data
            data = yf.download(self.tickers, 
                              start=self.start_date, 
                              end=self.end_date,
                              progress=True)
            
            # Extract different price types
            prices = {
                'adj_close': data['Adj Close'] if 'Adj Close' in data.columns else None,
                'close': data['Close'] if 'Close' in data.columns else None,
                'open': data['Open'] if 'Open' in data.columns else None,
                'high': data['High'] if 'High' in data.columns else None,
                'low': data['Low'] if 'Low' in data.columns else None,
                'volume': data['Volume'] if 'Volume' in data.columns else None
            }
            
            # Use Adjusted Close as primary price series
            if prices['adj_close'] is not None:
                main_prices = prices['adj_close']
            else:
                main_prices = prices['close']
            
            # Check for missing data
            missing_data = main_prices.isnull().sum()
            if missing_data.sum() > 0:
                print(f"\n!! Warning !!: Missing data detected:")
                for ticker in self.tickers:
                    if missing_data[ticker] > 0:
                        print(f"   {ticker}: {missing_data[ticker]} missing days")
                
                # Forward fill missing data
                main_prices = main_prices.fillna(method='ffill').fillna(method='bfill')
            
            # Save raw data
            for price_type, price_data in prices.items():
                if price_data is not None:
                    price_data.to_csv(f"{self.raw_dir}/{price_type}.csv")
            
            print(f"\nGood: Data downloaded successfully")
            print(f"   Shape: {main_prices.shape}")
            print(f"   Date range: {main_prices.index[0]} to {main_prices.index[-1]}")
            print(f"   Total trading days: {len(main_prices)}")
            
            return main_prices
            
        except Exception as e:
            print(f"\nERROR: Error downloading data: {str(e)}")
            return None
    
    def calculate_returns(self, prices):
        """
        Calculate various return metrics
        """
        print("\n" + "="*60)
        print("CALCULATING RETURN METRICS")
        print("="*60)
        
        # Simple returns
        simple_returns = prices.pct_change().dropna()
        
        # Log returns (continuously compounded)
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        # Rolling metrics
        rolling_window = 252  # One trading year
        
        rolling_stats = {}
        for ticker in self.tickers:
            rolling_stats[ticker] = {
                'rolling_vol': simple_returns[ticker].rolling(window=rolling_window).std() * np.sqrt(252),
                'rolling_mean': simple_returns[ticker].rolling(window=rolling_window).mean() * 252,
                'rolling_sharpe': (simple_returns[ticker].rolling(window=rolling_window).mean() * 252) / 
                                  (simple_returns[ticker].rolling(window=rolling_window).std() * np.sqrt(252))
            }
        
        # Calculate statistics
        stats = {
            'simple_returns': simple_returns,
            'log_returns': log_returns,
            'rolling_stats': rolling_stats,
            'summary': {
                'mean_returns': simple_returns.mean() * 252,
                'median_returns': simple_returns.median() * 252,
                'volatilities': simple_returns.std() * np.sqrt(252),
                'annualized_returns': (1 + simple_returns.mean())**252 - 1,
                'max_drawdown': self.calculate_max_drawdown(prices),
                'skewness': simple_returns.skew(),
                'kurtosis': simple_returns.kurtosis(),
                'jarque_bera': self.jarque_bera_test(simple_returns)
            },
            'correlation': {
                'pearson': simple_returns.corr(),
                'spearman': simple_returns.corr(method='spearman'),
                'rolling_corr': self.calculate_rolling_correlations(simple_returns)
            },
            'covariance': {
                'annualized': simple_returns.cov() * 252,
                'monthly': simple_returns.cov() * 21,
                'daily': simple_returns.cov()
            }
        }
        
        # Save returns data
        simple_returns.to_csv(f"{self.processed_dir}/simple_returns.csv")
        log_returns.to_csv(f"{self.processed_dir}/log_returns.csv")
        
        print(f"\nReturn Statistics:")
        print(f"   Mean Returns (ann): {stats['summary']['mean_returns'].values}")
        print(f"   Volatilities (ann): {stats['summary']['volatilities'].values}")
        print(f"   Max Drawdown: {stats['summary']['max_drawdown']}")
        
        return stats
    
    def calculate_max_drawdown(self, prices):
        """
        Calculate maximum drawdown for each asset
        """
        drawdowns = {}
        for ticker in self.tickers:
            wealth_index = prices[ticker] / prices[ticker].iloc[0]
            previous_peaks = wealth_index.cummax()
            drawdown = (wealth_index - previous_peaks) / previous_peaks
            drawdowns[ticker] = drawdown.min()
        
        return drawdowns
    
    def calculate_rolling_correlations(self, returns, window=60):
        """
        Calculate rolling correlations between assets
        """
        n_assets = len(self.tickers)
        rolling_corrs = {}
        
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                pair = f"{self.tickers[i]}_{self.tickers[j]}"
                rolling_corrs[pair] = returns[self.tickers[i]].rolling(window=window).corr(
                    returns[self.tickers[j]]
                )
        
        return pd.DataFrame(rolling_corrs)
    
    def jarque_bera_test(self, returns):
        """
        Perform Jarque-Bera test for normality
        """
        from scipy import stats
        
        results = {}
        for ticker in self.tickers:
            statistic, p_value = stats.jarque_bera(returns[ticker].dropna())
            results[ticker] = {
                'statistic': statistic,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }
        
        return results
    
    def calculate_var_cvar(self, returns, confidence_level=0.95):
        """
        Calculate Value at Risk and Conditional VaR
        """
        var_cvar = {}
        
        for ticker in self.tickers:
            ticker_returns = returns[ticker].dropna()
            
            # Historical VaR
            var_hist = np.percentile(ticker_returns, (1 - confidence_level) * 100)
            cvar_hist = ticker_returns[ticker_returns <= var_hist].mean()
            
            # Parametric VaR (assuming normal)
            from scipy.stats import norm
            z_score = norm.ppf(1 - confidence_level)
            var_param = ticker_returns.mean() + z_score * ticker_returns.std()
            
            var_cvar[ticker] = {
                'historical_var_95': var_hist,
                'historical_cvar_95': cvar_hist,
                'parametric_var_95': var_param,
                'confidence_level': confidence_level
            }
        
        return var_cvar
    
    def analyze_regime_changes(self, returns):
        """
        Analyze different market regimes
        """
        print("\n" + "="*60)
        print("MARKET REGIME ANALYSIS")
        print("="*60)
        
        # Split into different periods
        regimes = {
            'pre_covid': returns[:'2020-02-29'],
            'covid_crash': returns['2020-03-01':'2020-04-30'],
            'post_covid_recovery': returns['2020-05-01':'2021-12-31'],
            '2022_bear_market': returns['2022-01-01':'2022-12-31'],
            '2023_recovery': returns['2023-01-01':'2023-12-31'],
            '2024_consolidation': returns['2024-01-01':]
        }
        
        regime_stats = {}
        
        for regime_name, regime_returns in regimes.items():
            if len(regime_returns) > 0:
                regime_stats[regime_name] = {
                    'volatility': regime_returns.std() * np.sqrt(252),
                    'mean_return': regime_returns.mean() * 252,
                    'sharpe_ratio': (regime_returns.mean() * 252) / (regime_returns.std() * np.sqrt(252)),
                    'correlation': regime_returns.corr().values.tolist()
                }
                
                print(f"\n{regime_name.upper()}:")
                print(f"   Volatility: {regime_stats[regime_name]['volatility'].values}")
                print(f"   Mean Return: {regime_stats[regime_name]['mean_return'].values}")
        
        return regime_stats
    
    def create_visualizations(self, prices, returns, stats):
        """
        Create comprehensive visualizations
        """
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Price evolution
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Historical Price Analysis', fontsize=16, fontweight='bold')
        
        # Normalized prices
        normalized_prices = prices / prices.iloc[0] * 100
        normalized_prices.plot(ax=axes[0, 0])
        axes[0, 0].set_title('Normalized Prices (Base=100)')
        axes[0, 0].set_ylabel('Price Index')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Returns distribution
        for ticker in self.tickers:
            returns[ticker].hist(ax=axes[0, 1], bins=50, alpha=0.5, label=ticker)
        axes[0, 1].set_title('Returns Distribution')
        axes[0, 1].set_xlabel('Daily Returns')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rolling volatility
        for ticker in self.tickers:
            rolling_vol = returns[ticker].rolling(window=252).std() * np.sqrt(252)
            axes[1, 0].plot(rolling_vol.index, rolling_vol.values, label=ticker)
        axes[1, 0].set_title('Rolling 1-Year Volatility (Annualized)')
        axes[1, 0].set_ylabel('Volatility')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Correlation heatmap
        corr_matrix = stats['correlation']['pearson']
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Return Correlations')
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/price_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Additional plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Advanced Risk Metrics', fontsize=16, fontweight='bold')
        
        # Drawdown analysis
        wealth_index = prices / prices.iloc[0]
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        
        drawdowns.plot(ax=axes[0, 0])
        axes[0, 0].set_title('Drawdown Analysis')
        axes[0, 0].set_ylabel('Drawdown')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Rolling correlations
        rolling_corrs = stats['correlation']['rolling_corr']
        if rolling_corrs is not None:
            rolling_corrs.plot(ax=axes[0, 1])
            axes[0, 1].set_title('60-Day Rolling Correlations')
            axes[0, 1].set_ylabel('Correlation')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        # QQ plots for normality check
        from scipy import stats as scipy_stats
        for ticker in self.tickers:
            scipy_stats.probplot(returns[ticker].dropna(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normality Check)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Volatility clustering
        squared_returns = returns ** 2
        squared_returns.plot(ax=axes[1, 1])
        axes[1, 1].set_title('Squared Returns (Volatility Clustering)')
        axes[1, 1].set_ylabel('Squared Returns')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/risk_metrics.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Good: Visualizations saved to {self.plots_dir}/")
    
    def save_all_data(self, prices, returns_stats):
        """
        Save all processed data and statistics
        """
        print("\n" + "="*60)
        print("SAVING PROCESSED DATA")
        print("="*60)
        
        # Save prices
        prices.to_csv(f"{self.processed_dir}/prices.csv")
        
        # Save statistics
        stats_to_save = {
            'summary': returns_stats['summary'],
            'correlation': {
                'pearson': returns_stats['correlation']['pearson'].to_dict(),
                'spearman': returns_stats['correlation']['spearman'].to_dict()
            },
            'covariance': {
                'annualized': returns_stats['covariance']['annualized'].to_dict()
            }
        }
        
        with open(f"{self.processed_dir}/statistics.pkl", 'wb') as f:
            pickle.dump(stats_to_save, f)
        
        # Save VaR calculations
        var_cvar = self.calculate_var_cvar(returns_stats['simple_returns'])
        with open(f"{self.analysis_dir}/var_cvar.pkl", 'wb') as f:
            pickle.dump(var_cvar, f)
        
        # Save metadata
        metadata = {
            'download_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'tickers': self.tickers,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'n_observations': len(prices),
            'date_generated': datetime.now().isoformat()
        }
        
        with open(f"{self.processed_dir}/metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\nGood: All data saved to {self.output_dir}/")
        print(f"   - Raw data: {self.raw_dir}/")
        print(f"   - Processed data: {self.processed_dir}/")
        print(f"   - Analysis: {self.analysis_dir}/")
        print(f"   - Plots: {self.plots_dir}/")
        
        return metadata
    
    def generate_report(self, prices, returns_stats, metadata):
        """
        Generate a comprehensive text report
        """
        print("\n" + "="*60)
        print("GENERATING MARKET DATA REPORT")
        print("="*60)
        
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("BARRIER REVERSE CONVERTIBLE - HISTORICAL DATA REPORT")
        report_lines.append("="*60)
        report_lines.append(f"\nReport Generated: {metadata['download_date']}")
        report_lines.append(f"Data Period: {self.start_date} to {self.end_date}")
        report_lines.append(f"Trading Days: {metadata['n_observations']}")
        
        report_lines.append("\n" + "-"*40)
        report_lines.append("SUMMARY STATISTICS (Annualized)")
        report_lines.append("-"*40)
        
        for ticker in self.tickers:
            report_lines.append(f"\n{ticker}:")
            report_lines.append(f"   Mean Return: {returns_stats['summary']['mean_returns'][ticker]:.2%}")
            report_lines.append(f"   Volatility: {returns_stats['summary']['volatilities'][ticker]:.2%}")
            report_lines.append(f"   Max Drawdown: {returns_stats['summary']['max_drawdown'][ticker]:.2%}")
            report_lines.append(f"   Skewness: {returns_stats['summary']['skewness'][ticker]:.3f}")
            report_lines.append(f"   Kurtosis: {returns_stats['summary']['kurtosis'][ticker]:.3f}")
            
            # Normality test
            jb_result = returns_stats['summary']['jarque_bera'][ticker]
            report_lines.append(f"   Normal Distribution: {jb_result['is_normal']} (p={jb_result['p_value']:.3f})")
        
        report_lines.append("\n" + "-"*40)
        report_lines.append("CORRELATION MATRIX")
        report_lines.append("-"*40)
        
        corr_matrix = returns_stats['correlation']['pearson']
        report_lines.append("\n" + corr_matrix.round(3).to_string())
        
        report_lines.append("\n" + "-"*40)
        report_lines.append("COVARIANCE MATRIX (Annualized)")
        report_lines.append("-"*40)
        
        cov_matrix = returns_stats['covariance']['annualized']
        report_lines.append("\n" + cov_matrix.round(4).to_string())
        
        # Write report to file
        with open(f"{self.analysis_dir}/data_report.txt", 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Print report
        print('\n'.join(report_lines))
        
        print(f"\nGood: Report saved to {self.analysis_dir}/data_report.txt")
    
    def run_full_download(self):
        """
        Execute complete data download and processing pipeline
        """
        print("\n" + "="*60)
        print("HISTORICAL DATA DOWNLOADER - STARTED")
        print("="*60)
        print(f"Target Tickers: {self.tickers}")
        print(f"Output Directory: {self.output_dir}")
        
        # Step 1: Download data
        prices = self.download_data()
        if prices is None:
            print("ERROR: Data download failed. Exiting.")
            return None
        
        # Step 2: Calculate returns and statistics
        returns_stats = self.calculate_returns(prices)
        
        # Step 3: Analyze regime changes
        regimes = self.analyze_regime_changes(returns_stats['simple_returns'])
        
        # Step 4: Create visualizations
        self.create_visualizations(prices, returns_stats['simple_returns'], returns_stats)
        
        # Step 5: Save all data
        metadata = self.save_all_data(prices, returns_stats)
        
        # Step 6: Generate report
        self.generate_report(prices, returns_stats, metadata)
        
        print("\n" + "="*60)
        print("DATA DOWNLOAD COMPLETE")
        print("="*60)
        print(f"All data saved to: {self.output_dir}/")
        
        return {
            'prices': prices,
            'returns': returns_stats['simple_returns'],
            'log_returns': returns_stats['log_returns'],
            'statistics': returns_stats,
            'regimes': regimes,
            'metadata': metadata
        }

def main():
    """Main execution function"""
    
    # Configuration
    tickers = ['NFLX', 'SPOT', 'DIS']
    start_date = '2020-01-01'
    end_date = '2025-04-01'
    
    # Initialize downloader
    downloader = HistoricalDataDownloader(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        output_dir='../../final_dataset/prices/fetch_prices_v1'
    )
    
    # Run full download pipeline
    results = downloader.run_full_download()
    
    if results:
        print("\n" + "="*60)
        print("QUICK ACCESS - KEY DATASETS")
        print("="*60)
        print("\nTo load the data in main script:")
        print("\n# Load prices")
        print("prices = pd.read_csv('./historical_data/processed/prices.csv', index_col=0)")
        print("\n# Load returns")
        print("returns = pd.read_csv('./historical_data/processed/simple_returns.csv', index_col=0)")
        print("\n# Load statistics")
        print("import pickle")
        print("with open('./historical_data/processed/statistics.pkl', 'rb') as f:")
        print("    stats = pickle.load(f)")
    else:
        print("\nERROR: Data download failed. Check internet connection and ticker symbols.")

if __name__ == "__main__":
    main()

    