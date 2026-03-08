"""
Black-Scholes Synthetic Option Data Generator
Generates arbitrage-free option surfaces using Black-Scholes with term structure
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

class BlackScholesSyntheticGenerator:
    """
    Generate synthetic option data using Black-Scholes with term structure
    and volatility smile parameterization
    """
    
    def __init__(self, tickers=['NFLX', 'SPOT', 'DIS'], 
                 output_dir='../../final_dataset/synthetic_option/black_scholes'):
        
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.output_dir = output_dir
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Market parameters
        self.risk_free_rate = 0.05  # Constant rate for simplicity
        
        # Spot prices (as of example)
        self.spot_prices = {
            'NFLX': 850.0,
            'SPOT': 350.0,
            'DIS': 110.0
        }
        
    def volatility_term_structure(self, T, base_vol, vol_of_vol=0.1, mean_reversion=2.0):
        """
        Generate term structure of volatility
        Longer maturities have mean-reverting volatility
        """
        # Vasicek-like term structure
        if T < 0.1:
            return base_vol
        else:
            # Volatility converges to long-term mean
            long_term_vol = base_vol * 1.2
            return long_term_vol + (base_vol - long_term_vol) * np.exp(-mean_reversion * T)
    
    def volatility_smile(self, moneyness, T, atm_vol, skew=0.2, smile=0.1):
        """
        Generate volatility smile using quadratic parameterization
        volatility = atm_vol + skew * (moneyness - 1) + smile * (moneyness - 1)^2
        """
        return atm_vol + skew * (moneyness - 1) + smile * (moneyness - 1)**2
    
    def black_scholes_call(self, S, K, T, r, sigma):
        """Black-Scholes call price"""
        if T <= 0 or sigma <= 0:
            return max(0, S - K)
        
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    def black_scholes_put(self, S, K, T, r, sigma):
        """Black-Scholes put price via put-call parity"""
        call_price = self.black_scholes_call(S, K, T, r, sigma)
        return call_price - S + K * np.exp(-r * T)
    
    def implied_volatility(self, price, S, K, T, r, option_type='call'):
        """
        Calculate implied volatility from option price
        """
        if T <= 0:
            return np.nan
            
        def objective(sigma):
            if option_type == 'call':
                return self.black_scholes_call(S, K, T, r, sigma) - price
            else:
                return self.black_scholes_put(S, K, T, r, sigma) - price
        
        try:
            return brentq(objective, 0.001, 2.0)
        except:
            return np.nan
    
    def generate_volatility_surface_params(self, ticker):
        """
        Generate realistic volatility surface parameters for each ticker
        Different tickers have different volatility characteristics
        """
        if ticker == 'NFLX':
            # High growth stock - higher vol, more skew
            return {
                'base_vol': 0.35,      # Higher volatility
                'vol_of_vol': 0.15,
                'mean_reversion': 2.0,
                'skew': -0.3,           # Negative skew (typical for equities)
                'smile': 0.15
            }
        elif ticker == 'SPOT':
            # Tech stock - moderate vol, moderate skew
            return {
                'base_vol': 0.30,
                'vol_of_vol': 0.12,
                'mean_reversion': 1.8,
                'skew': -0.25,
                'smile': 0.12
            }
        else:  # DIS
            # Mature stock - lower vol, less skew
            return {
                'base_vol': 0.25,
                'vol_of_vol': 0.10,
                'mean_reversion': 1.5,
                'skew': -0.20,
                'smile': 0.10
            }
    
    def generate_option_surface(self, ticker):
        """
        Generate complete option surface for a single ticker
        """
        print(f"\nGenerating option surface for {ticker}...")
        
        # Parameters
        S0 = self.spot_prices[ticker]
        vol_params = self.generate_volatility_surface_params(ticker)
        
        # Grid specification
        maturities = np.array([0.0833, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])  # 1mo to 2yr
        moneyness = np.array([0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3])
        
        option_data = []
        
        for T in maturities:
            # Get term structure adjusted base volatility
            atm_vol = self.volatility_term_structure(
                T, 
                vol_params['base_vol'],
                vol_params['vol_of_vol'],
                vol_params['mean_reversion']
            )
            
            for m in moneyness:
                K = S0 * m
                
                # Apply volatility smile
                sigma = self.volatility_smile(
                    m, T, atm_vol,
                    vol_params['skew'],
                    vol_params['smile']
                )
                
                # Ensure volatility is positive and reasonable
                sigma = max(0.05, min(1.0, sigma))
                
                # Generate call and put prices
                call_price = self.black_scholes_call(S0, K, T, self.risk_free_rate, sigma)
                put_price = self.black_scholes_put(S0, K, T, self.risk_free_rate, sigma)
                
                # Calculate Greeks
                d1 = (np.log(S0/K) + (self.risk_free_rate + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                
                delta_call = norm.cdf(d1)
                gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
                vega = S0 * norm.pdf(d1) * np.sqrt(T)
                theta_call = - (S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - \
                             self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * norm.cdf(d2)
                
                option_data.append({
                    'ticker': ticker,
                    'valuation_date': '2025-04-01',
                    'maturity_years': T,
                    'maturity_days': int(T * 365),
                    'strike': K,
                    'moneyness': m,
                    'spot_price': S0,
                    'call_price': call_price,
                    'put_price': put_price,
                    'implied_vol': sigma,
                    'delta_call': delta_call,
                    'gamma': gamma,
                    'vega': vega,
                    'theta_call': theta_call,
                    'risk_free_rate': self.risk_free_rate
                })
        
        df = pd.DataFrame(option_data)
        
        # Save individual ticker data
        df.to_csv(f"{self.output_dir}/{ticker}_bs_surface.csv", index=False)
        print(f"   Generated {len(df)} options for {ticker}")
        
        return df
    

    def prepare_calibration_data(self, option_data, historical_returns=None):
        """
        Prepare organized data structures for model calibration
        
        This function transforms raw option data into model-specific formats
        that calibration algorithms can easily consume.
        """
        print("\n" + "="*60)
        print("PREPARING CALIBRATION DATASETS")
        print("="*60)
        
        calibration_data = {
            'metadata': {
                'valuation_date': '2025-04-01',
                'risk_free_rate': self.risk_free_rate,
                'n_tickers': len(self.tickers)
            },
            'heston': {},
            'bs_smile': {},  # For Black-Scholes with smile
            'surface_stats': {},
            'validation': {}
        }
        
        for ticker in self.tickers:
            ticker_options = option_data[option_data['ticker'] == ticker].copy()
            S0 = ticker_options['spot_price'].iloc[0]
            
            # === HESTON CALIBRATION DATA ===
            # Need full surface with clean structure for numerical optimization
            heston_data = {
                'spot': S0,
                'maturities': sorted(ticker_options['maturity_years'].unique()),
                'strikes': {},
                'prices': {},
                'implied_vols': {},
                'weights': {}  # For calibration weighting
            }
            
            for T in heston_data['maturities']:
                T_data = ticker_options[ticker_options['maturity_years'] == T]
                T_data = T_data.sort_values('strike')
                
                heston_data['strikes'][T] = T_data['strike'].values.tolist()
                heston_data['prices'][T] = T_data['call_price'].values.tolist()
                heston_data['implied_vols'][T] = T_data['implied_vol'].values.tolist()
                
                # Vega-weighted calibration (options with higher vega get more weight)
                vega = self.calculate_vega(S0, T_data['strike'].values, T, 
                                        self.risk_free_rate, T_data['implied_vol'].values)
                heston_data['weights'][T] = (vega / vega.sum()).tolist()
            
            calibration_data['heston'][ticker] = heston_data
            
            # === BLACK-SCHOLES WITH SMILE CALIBRATION ===
            # For models that parameterize the volatility surface
            bs_smile_data = {
                'spot': S0,
                'surface': self.extract_vol_surface(ticker_options),
                'term_structure': ticker_options.groupby('maturity_years')['implied_vol'].agg(['mean', 'std']).to_dict(),
                'smile_params': self.fit_initial_smile(ticker_options)  # Quick initial guess
            }
            calibration_data['bs_smile'][ticker] = bs_smile_data
            
            # === SURFACE STATISTICS ===
            # Useful for diagnostics and validation
            surface_stats = {
                'min_vol': ticker_options['implied_vol'].min(),
                'max_vol': ticker_options['implied_vol'].max(),
                'atm_vols': ticker_options[ticker_options['moneyness'] == 1.0]
                        .groupby('maturity_years')['implied_vol'].mean().to_dict(),
                'skew': self.calculate_skew(ticker_options),
                'convexity': self.calculate_convexity(ticker_options)
            }
            calibration_data['surface_stats'][ticker] = surface_stats
            
            # === VALIDATION METRICS ===
            # Check data quality before calibration
            validation = {
                'has_negative_prices': (ticker_options['call_price'] < 0).any(),
                'has_nan_vols': ticker_options['implied_vol'].isna().any(),
                'arbitrage_violations': self.check_arbitrage(ticker_options),
                'n_options': len(ticker_options),
                'n_maturities': len(heston_data['maturities'])
            }
            calibration_data['validation'][ticker] = validation
            
            print(f"\n{ticker} Calibration Data:")
            print(f"   - Options: {validation['n_options']}")
            print(f"   - Maturities: {validation['n_maturities']}")
            print(f"   - ATM Vol Range: {min(surface_stats['atm_vols'].values()):.2f} - "
                f"{max(surface_stats['atm_vols'].values()):.2f}")
            print(f"   - Skew (90-110): {surface_stats['skew']:.3f}")
        
        # Save organized data
        with open(f"{self.output_dir}/calibration_data.pkl", 'wb') as f:
            pickle.dump(calibration_data, f)
        
        return calibration_data

    def calculate_vega(self, S, K, T, r, sigma):
        """Calculate vega for weighting options in calibration"""
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T)

    def extract_vol_surface(self, df):
        """Extract volatility surface as matrix for easy plotting/analysis"""
        maturities = sorted(df['maturity_years'].unique())
        moneyness = sorted(df['moneyness'].unique())
        
        surface = np.zeros((len(maturities), len(moneyness)))
        for i, T in enumerate(maturities):
            for j, m in enumerate(moneyness):
                val = df[(df['maturity_years'] == T) & (df['moneyness'] == m)]['implied_vol'].values
                surface[i, j] = val[0] if len(val) > 0 else np.nan
        
        return {
            'maturities': maturities,
            'moneyness': moneyness,
            'surface': surface
        }

    def calculate_skew(self, df):
        """Calculate volatility skew (90% moneyness vol - 110% moneyness vol)"""
        atm = df[df['moneyness'] == 1.0].groupby('maturity_years')['implied_vol'].mean()
        put_skew = df[df['moneyness'] == 0.9].groupby('maturity_years')['implied_vol'].mean()
        call_skew = df[df['moneyness'] == 1.1].groupby('maturity_years')['implied_vol'].mean()
        
        # Average skew across maturities
        skew = (put_skew - call_skew).mean()
        return skew

    def calculate_convexity(self, df):
        """Calculate smile convexity (average of (vol_90 + vol_110 - 2*vol_100))"""
        vols_90 = df[df['moneyness'] == 0.9].groupby('maturity_years')['implied_vol'].mean()
        vols_100 = df[df['moneyness'] == 1.0].groupby('maturity_years')['implied_vol'].mean()
        vols_110 = df[df['moneyness'] == 1.1].groupby('maturity_years')['implied_vol'].mean()
        
        convexity = (vols_90 + vols_110 - 2 * vols_100).mean()
        return convexity

    def fit_initial_smile(self, df):
        """Quick initial parameter guess for smile calibration"""
        # Fit quadratic: vol = a + b*(m-1) + c*(m-1)^2
        from scipy.optimize import curve_fit
        
        def smile_func(m, a, b, c):
            return a + b*(m-1) + c*(m-1)**2
        
        initial_params = {}
        for T in df['maturity_years'].unique():
            T_data = df[df['maturity_years'] == T]
            try:
                popt, _ = curve_fit(smile_func, T_data['moneyness'], T_data['implied_vol'])
                initial_params[T] = {'a': popt[0], 'b': popt[1], 'c': popt[2]}
            except:
                initial_params[T] = {'a': 0.3, 'b': -0.1, 'c': 0.05}
        
        return initial_params

    def check_arbitrage(self, df):
        """Basic arbitrage checks"""
        violations = []
        
        for T in df['maturity_years'].unique():
            T_data = df[df['maturity_years'] == T].sort_values('strike')
            prices = T_data['call_price'].values
            
            # Check monotonicity
            if not np.all(np.diff(prices) <= 1e-6):
                violations.append(f"T={T}: Non-monotonic prices")
            
            # Check convexity
            if len(prices) >= 3:
                second_diff = np.diff(prices, 2)
                if not np.all(second_diff >= -1e-6):
                    violations.append(f"T={T}: Non-convex prices")
        
        return violations
    

    def generate_all_surfaces(self):
        """
        Generate option surfaces for all tickers
        """
        print("\n" + "="*60)
        print("GENERATING BLACK-SCHOLES SYNTHETIC OPTION DATA")
        print("="*60)
        
        all_data = []
        surface_params = {}
        
        for ticker in self.tickers:
            df = self.generate_option_surface(ticker)
            all_data.append(df)
            
            # Store surface parameters
            surface_params[ticker] = self.generate_volatility_surface_params(ticker)
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(f"{self.output_dir}/all_bs_surfaces.csv", index=False)
        
        # Save parameters
        with open(f"{self.output_dir}/bs_surface_params.pkl", 'wb') as f:
            pickle.dump(surface_params, f)
        
        # Prepare calibration data
        calibration_data = self.prepare_calibration_data(combined_df)
        # calibration_data --> Saved to pickle file f"{self.output_dir}/calibration_data.pkl"

        print("\n" + "="*60)
        print("GENERATION COMPLETE")
        print("="*60)
        print(f"Total options generated: {len(combined_df)}")
        print(f"Files saved to: {self.output_dir}/")
        print("   - [TICKER]_bs_surface.csv (individual surfaces)")
        print("   - all_bs_surfaces.csv (combined data)")
        print("   - bs_surface_params.pkl (surface parameters)")
        print("   - calibration_data.csv (for modelling)")
        
        return combined_df, surface_params
    
    def validate_surface(self, df):
        """
        Validate that the generated surface is arbitrage-free
        """
        print("\n" + "="*60)
        print("VALIDATING OPTION SURFACE")
        print("="*60)
        
        validation_results = {}
        
        for ticker in self.tickers:
            ticker_data = df[df['ticker'] == ticker]
            
            # Check 1: Call prices decrease with strike (for same maturity)
            for T in ticker_data['maturity_years'].unique():
                mask = ticker_data['maturity_years'] == T
                T_data = ticker_data[mask].sort_values('strike')
                
                # Check monotonicity
                call_prices = T_data['call_price'].values
                is_monotonic = np.all(np.diff(call_prices) <= 0)
                
                # Check convexity
                if len(call_prices) >= 3:
                    is_convex = np.all(np.diff(call_prices, 2) >= 0)
                else:
                    is_convex = True
                
                validation_results[f"{ticker}_T{T}"] = {
                    'monotonic': is_monotonic,
                    'convex': is_convex
                }
            
            # Check 2: Put-call parity
            sample = ticker_data[ticker_data['moneyness'] == 1.0].iloc[0]
            S0 = sample['spot_price']
            K = sample['strike']
            T = sample['maturity_years']
            call = sample['call_price']
            put = sample['put_price']
            
            parity_check = abs((call - put) - (S0 - K * np.exp(-self.risk_free_rate * T)))
            validation_results[f"{ticker}_put_call_parity"] = parity_check < 0.01
        
        # Print validation results
        for key, value in validation_results.items():
            print(f"   {key}: {value}")
        
        return validation_results

def main():
    """Main execution function"""
    # Initialize generator
    generator = BlackScholesSyntheticGenerator(
        tickers=['NFLX', 'SPOT', 'DIS'],
        output_dir='../../final_dataset/synthetic_option/black_scholes'
    )
    
    # Generate option surfaces
    option_data, params = generator.generate_all_surfaces()
    
    # Validate surfaces
    validation = generator.validate_surface(option_data)
    
    # Display sample data
    print("\n" + "="*60)
    print("SAMPLE OPTION DATA (ATM Options)")
    print("="*60)
    sample = option_data[option_data['moneyness'] == 1.0].head(10)
    print(sample[['ticker', 'maturity_years', 'strike', 'call_price', 
                  'put_price', 'implied_vol', 'delta_call']].to_string())

if __name__ == "__main__":
    main()

    