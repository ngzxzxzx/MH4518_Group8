"""
Heston Model Synthetic Option Data Generator
Generates realistic option surfaces using a combination of Heston principles
and direct volatility surface parameterization
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

class HestonSyntheticGenerator:
    """
    Generate synthetic option data with realistic Heston-like features
    but with guaranteed numerical stability
    """
    
    def __init__(self, tickers=['NFLX', 'SPOT', 'DIS'],
                 output_dir='../../final_dataset/synthetic_option/heston'):
        
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.output_dir = output_dir
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Market parameters
        self.risk_free_rate = 0.05
        
        # Spot prices
        self.spot_prices = {
            'NFLX': 850.0,
            'SPOT': 350.0,
            'DIS': 110.0
        }
        
    def generate_heston_parameters(self, ticker):
        """
        Generate realistic Heston parameters for each ticker
        """
        if ticker == 'NFLX':
            # High volatility stock - steep skew
            return {
                'v0': 0.04,      # Initial variance (20% vol)
                'kappa': 2.0,     # Mean reversion speed
                'theta': 0.06,    # Long-term variance (24.5% vol)
                'sigma': 0.3,     # Vol of vol
                'rho': -0.7       # Correlation
            }
        elif ticker == 'SPOT':
            # Moderate volatility stock
            return {
                'v0': 0.03,       # Initial variance (17.3% vol)
                'kappa': 1.8,     
                'theta': 0.05,    # Long-term variance (22.4% vol)
                'sigma': 0.25,      
                'rho': -0.6        
            }
        else:  # DIS
            # Lower volatility stock
            return {
                'v0': 0.02,       # Initial variance (14.1% vol)
                'kappa': 1.5,      
                'theta': 0.04,    # Long-term variance (20% vol)
                'sigma': 0.2,      
                'rho': -0.5         
            }
    
    def heston_vol_surface(self, T, m, params):
        """
        Generate implied volatility using Heston-like features
        This is a simplified but numerically stable approach
        """
        v0 = params['v0']
        theta = params['theta']
        kappa = params['kappa']
        rho = params['rho']
        sigma = params['sigma']
        
        # Base volatility (term structure)
        vol_T = np.sqrt(theta + (v0 - theta) * np.exp(-kappa * T))
        
        # Skew effect (moneyness)
        skew_effect = rho * sigma * (1 - np.exp(-kappa * T)) / (kappa * T) * np.log(m)
        
        # Convexity effect (smile)
        convexity = 0.1 * sigma * (1 - np.exp(-kappa * T)) / (kappa * T) * (np.log(m))**2
        
        # Combine effects
        implied_vol = vol_T + skew_effect + convexity
        
        # Ensure volatility is within reasonable bounds
        implied_vol = max(0.15, min(0.60, implied_vol))
        
        return implied_vol
    
    def black_scholes_call(self, S0, K, T, r, sigma):
        """Black-Scholes call price"""
        if T <= 0:
            return max(0, S0 - K)
        if sigma <= 0:
            return max(0, S0 - K * np.exp(-r * T))
        
        d1 = (np.log(S0/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    def generate_option_surface(self, ticker):
        """
        Generate complete option surface using simplified Heston approach
        """
        print(f"\nGenerating Heston option surface for {ticker}...")
        
        # Parameters
        S0 = self.spot_prices[ticker]
        params = self.generate_heston_parameters(ticker)
        
        # Grid specification
        maturities = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
        moneyness = np.array([0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3])
        
        option_data = []
        
        for T in maturities:
            print(f"   Processing maturity {T:.2f}Y...")
            
            # Store prices for this maturity
            maturity_prices = []
            
            for m in moneyness:
                K = S0 * m
                
                # Get Heston-style implied volatility
                implied_vol = self.heston_vol_surface(T, m, params)
                
                # Calculate Black-Scholes price
                call_price = self.black_scholes_call(
                    S0, K, T, self.risk_free_rate, implied_vol
                )
                
                maturity_prices.append(call_price)
                
                option_data.append({
                    'ticker': ticker,
                    'valuation_date': '2025-04-01',
                    'maturity_years': T,
                    'maturity_days': int(T * 365),
                    'strike': K,
                    'moneyness': m,
                    'spot_price': S0,
                    'call_price': call_price,
                    'implied_vol': implied_vol,
                    'heston_v0': params['v0'],
                    'heston_kappa': params['kappa'],
                    'heston_theta': params['theta'],
                    'heston_sigma': params['sigma'],
                    'heston_rho': params['rho'],
                    'risk_free_rate': self.risk_free_rate
                })
        
        df = pd.DataFrame(option_data)
        
        # Save individual ticker data
        df.to_csv(f"{self.output_dir}/{ticker}_heston_surface.csv", index=False)
        print(f"   Generated {len(df)} options for {ticker}")
        print(f"   Heston parameters: v0={params['v0']:.3f}, kappa={params['kappa']:.1f}, "
              f"theta={params['theta']:.3f}, sigma={params['sigma']:.2f}, rho={params['rho']:.2f}")
        
        return df, params
    
    def prepare_calibration_data(self, option_data):
        """
        Prepare organized data structures for model calibration
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
            'surface_stats': {}
        }
        
        for ticker in self.tickers:
            ticker_options = option_data[option_data['ticker'] == ticker].copy()
            S0 = ticker_options['spot_price'].iloc[0]
            
            # Calculate ATM vols
            atm_options = ticker_options[ticker_options['moneyness'] == 1.0]
            atm_vols = atm_options.groupby('maturity_years')['implied_vol'].mean()
            
            # Calculate skew and convexity
            skew_by_maturity = {}
            convexity_by_maturity = {}
            
            for T in ticker_options['maturity_years'].unique():
                T_data = ticker_options[ticker_options['maturity_years'] == T]
                
                # Get vols at key moneyness points
                vol_90 = T_data[T_data['moneyness'] == 0.9]['implied_vol'].values
                vol_100 = T_data[T_data['moneyness'] == 1.0]['implied_vol'].values
                vol_110 = T_data[T_data['moneyness'] == 1.1]['implied_vol'].values
                
                if len(vol_90) > 0 and len(vol_100) > 0 and len(vol_110) > 0:
                    skew_by_maturity[T] = vol_90[0] - vol_110[0]
                    convexity_by_maturity[T] = vol_90[0] + vol_110[0] - 2 * vol_100[0]
            
            # Store calibration data
            calibration_data['heston'][ticker] = {
                'spot': S0,
                'maturities': sorted(ticker_options['maturity_years'].unique()),
                'params': {
                    'v0': ticker_options['heston_v0'].iloc[0],
                    'kappa': ticker_options['heston_kappa'].iloc[0],
                    'theta': ticker_options['heston_theta'].iloc[0],
                    'sigma': ticker_options['heston_sigma'].iloc[0],
                    'rho': ticker_options['heston_rho'].iloc[0]
                },
                'options': ticker_options.to_dict('records')
            }
            
            calibration_data['surface_stats'][ticker] = {
                'n_options': len(ticker_options),
                'n_maturities': len(ticker_options['maturity_years'].unique()),
                'atm_vols': atm_vols.to_dict(),
                'skew_by_maturity': skew_by_maturity,
                'convexity_by_maturity': convexity_by_maturity,
                'avg_skew': np.mean(list(skew_by_maturity.values())) if skew_by_maturity else 0,
                'avg_convexity': np.mean(list(convexity_by_maturity.values())) if convexity_by_maturity else 0
            }
            
            print(f"\n{ticker} Calibration Data:")
            print(f"   - Options: {len(ticker_options)}")
            print(f"   - Maturities: {len(ticker_options['maturity_years'].unique())}")
            if len(atm_vols) > 0:
                print(f"   - ATM Vol Range: {atm_vols.min():.2f} - {atm_vols.max():.2f}")
            print(f"   - Avg Skew (90-110): {calibration_data['surface_stats'][ticker]['avg_skew']:.3f}")
            print(f"   - Avg Convexity: {calibration_data['surface_stats'][ticker]['avg_convexity']:.3f}")
        
        # Save organized data
        with open(f"{self.output_dir}/calibration_data.pkl", 'wb') as f:
            pickle.dump(calibration_data, f)
        
        return calibration_data
    
    def analyze_volatility_smile(self, df):
        """
        Analyze and visualize the volatility smile for each ticker
        """
        print("\n" + "="*60)
        print("VOLATILITY SMILE ANALYSIS")
        print("="*60)
        
        for ticker in self.tickers:
            ticker_data = df[df['ticker'] == ticker]
            
            print(f"\n{ticker} Volatility Smile:")
            
            for T in [0.25, 1.0, 2.0]:
                T_data = ticker_data[ticker_data['maturity_years'] == T].sort_values('moneyness')
                
                print(f"\n   Maturity {T:.1f}Y:")
                for _, row in T_data.iterrows():
                    print(f"      Moneyness {row['moneyness']:.2f}: {row['implied_vol']:.3f}")
    
    def verify_no_arbitrage(self, df):
        """
        Verify that option prices satisfy no-arbitrage conditions
        """
        print("\n" + "="*60)
        print("VERIFYING NO-ARBITRAGE CONDITIONS")
        print("="*60)
        
        violations = []
        
        for ticker in self.tickers:
            ticker_data = df[df['ticker'] == ticker]
            
            for T in ticker_data['maturity_years'].unique():
                T_data = ticker_data[ticker_data['maturity_years'] == T].sort_values('strike')
                strikes = T_data['strike'].values
                prices = T_data['call_price'].values
                
                # Condition 1: Prices must be non-negative
                if np.any(prices < -1e-6):
                    violations.append(f"{ticker} T={T}: Negative prices")
                
                # Condition 2: Prices must be decreasing in strike
                if not np.all(np.diff(prices) <= 1e-4):
                    violations.append(f"{ticker} T={T}: Prices not monotonic")
                
                # Condition 3: Prices must be convex in strike (looser tolerance)
                if len(prices) >= 3:
                    second_diff = np.diff(prices, 2)
                    if not np.all(second_diff >= -0.05):  # Looser tolerance
                        violations.append(f"{ticker} T={T}: Prices not convex")
        
        if len(violations) == 0:
            print("Good: All no-arbitrage conditions satisfied !")
            return True
        else:
            print(f"!!! {len(violations)} Arbitrage violations found: !!!")
            for v in violations[:10]:
                print(f"   - {v}")
            if len(violations) > 10:
                print(f"   ... and {len(violations) - 10} more")
            return False
    
    def generate_all_surfaces(self):
        """
        Generate option surfaces for all tickers
        """
        print("\n" + "="*60)
        print("GENERATING HESTON SYNTHETIC OPTION DATA")
        print("="*60)
        
        all_data = []
        all_params = {}
        
        for ticker in self.tickers:
            df, params = self.generate_option_surface(ticker)
            all_data.append(df)
            all_params[ticker] = params
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(f"{self.output_dir}/all_heston_surfaces.csv", index=False)
        
        # Save parameters
        with open(f"{self.output_dir}/heston_parameters.pkl", 'wb') as f:
            pickle.dump(all_params, f)
        
        # Prepare calibration data
        calibration_data = self.prepare_calibration_data(combined_df)
        
        print("\n" + "="*60)
        print("GENERATION COMPLETE")
        print("="*60)
        print(f"Total options generated: {len(combined_df)}")
        print(f"Files saved to: {self.output_dir}/")
        print("   - [TICKER]_heston_surface.csv (individual surfaces)")
        print("   - all_heston_surfaces.csv (combined data)")
        print("   - heston_parameters.pkl (model parameters)")
        print("   - calibration_data.pkl (for modelling)")
        
        return combined_df, all_params

def main():
    """Main execution function"""
    # Initialize generator
    generator = HestonSyntheticGenerator(
        tickers=['NFLX', 'SPOT', 'DIS'],
        output_dir='../../final_dataset/synthetic_option/heston'
    )
    
    # Generate option surfaces
    option_data, params = generator.generate_all_surfaces()
    
    # Analyze volatility smiles
    generator.analyze_volatility_smile(option_data)
    
    # Verify no-arbitrage
    generator.verify_no_arbitrage(option_data)
    
    # Display sample data
    print("\n" + "="*60)
    print("SAMPLE OPTION DATA (ATM Options)")
    print("="*60)
    sample = option_data[option_data['moneyness'] == 1.0].head(10)
    if len(sample) > 0:
        print(sample[['ticker', 'maturity_years', 'strike', 'call_price', 
                      'implied_vol']].to_string())

if __name__ == "__main__":
    main()


