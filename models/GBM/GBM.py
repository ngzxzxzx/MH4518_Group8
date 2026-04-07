"""
Geometric Brownian Motion (GBM) Model for Barrier Reverse Convertible Pricing
Enhanced version using actual historical and synthetic option data
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq, minimize
from datetime import datetime, timedelta
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class GBMBarrierReverseConvertible:
    """
    GBM Model for pricing Barrier Reverse Convertibles with Monte Carlo simulation
    Using actual calibrated parameters from historical and option data
    """
    
    def __init__(self,
                 tickers: List[str] = ['NFLX', 'SPOT', 'DIS'],
                 valuation_date: str = '2025-04-01',
                 maturity_date: str = '2026-10-01',
                 barrier_level: float = 0.50,  # 50% barrier
                 coupon_rate: float = 0.1375,  # 13.75% p.a.
                 principal: float = 100.0,
                 n_simulations: int = 1000,
                 n_steps: int = 378,  # ~18 months of trading days
                 data_dir: str = '../../final_dataset'):
        """
        Initialize GBM model for BRC pricing with actual data
        """
        
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.valuation_date = pd.to_datetime(valuation_date)
        self.maturity_date = pd.to_datetime(maturity_date)
        self.barrier_level = barrier_level
        self.coupon_rate = coupon_rate
        self.principal = principal
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.data_dir = data_dir
        
        # Calculate time parameters
        self.T = (self.maturity_date - self.valuation_date).days / 365.0
        self.dt = self.T / self.n_steps
        self.payment_dates = self._generate_payment_dates()
        
        # Initialize storage for model parameters
        self.spot_prices = {}
        self.drift_rates = {}
        self.volatilities = {}
        self.correlation_matrix = None
        self.cholesky_matrix = None
        
        # Load and calibrate all data
        self._load_and_calibrate_data()
        
        # Setup output directory
        self.output_dir = "./results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            os.makedirs(f"{self.output_dir}/simulations")
            os.makedirs(f"{self.output_dir}/results")
            os.makedirs(f"{self.output_dir}/plots")
    
    def _generate_payment_dates(self) -> List[datetime]:
        """Generate quarterly coupon payment dates"""
        payment_dates = []
        current_date = self.valuation_date
        
        while current_date < self.maturity_date:
            current_date = current_date + pd.DateOffset(months=3)
            if current_date < self.maturity_date:
                payment_dates.append(current_date)
        
        # Add maturity date
        payment_dates.append(self.maturity_date)
        
        print(f"\nCoupon Payment Dates ({len(payment_dates)} payments):")
        for i, date in enumerate(payment_dates[:-1], 1):
            print(f"   Payment {i}: {date.strftime('%Y-%m-%d')}")
        print(f"   Maturity: {payment_dates[-1].strftime('%Y-%m-%d')}")
        
        return payment_dates
    
    def _load_and_calibrate_data(self):
        """Load historical and option data for calibration"""
        print("\n" + "="*60)
        print("CALIBRATING GBM MODEL WITH ACTUAL DATA")
        print("="*60)
        
        # Load historical data
        self._load_historical_data()
        
        # Load option-implied data and calibrate
        self._load_option_data()
        
        # Blend historical and implied parameters
        self._blend_parameters()
        
        # Calculate Cholesky decomposition
        self.cholesky_matrix = np.linalg.cholesky(self.correlation_matrix)
        
        print("\n" + "-"*40)
        print("FINAL CALIBRATED PARAMETERS")
        print("-"*40)
        print(f"\nSpot Prices:")
        for ticker, price in self.spot_prices.items():
            print(f"   {ticker}: ${price:.2f}")
        
        print(f"\nDrift Rates (Risk-neutral):")
        for ticker, drift in self.drift_rates.items():
            print(f"   {ticker}: {drift:.2%}")
        
        print(f"\nVolatilities (Annualized):")
        for ticker, vol in self.volatilities.items():
            print(f"   {ticker}: {vol:.2%}")
        
        print(f"\nCorrelation Matrix:")
        corr_df = pd.DataFrame(self.correlation_matrix, 
                              index=self.tickers, columns=self.tickers)
        print(corr_df.round(3))
    
    def _load_historical_data(self):
        """Load historical price data from fetch_prices_v1 output"""
        print("\n--- Loading Historical Data ---")
        
        try:
            # Load statistics from pickle
            stats_path = f"{self.data_dir}/prices/fetch_prices_v1/processed/statistics.pkl"
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
            
            # Load prices to get spot prices
            prices_path = f"{self.data_dir}/prices/fetch_prices_v1/processed/prices.csv"
            prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
            
            # Extract spot prices (last available)
            for ticker in self.tickers:
                self.spot_prices[ticker] = prices[ticker].iloc[-1]
            
            # Extract historical volatilities
            # From output: 
            # NFLX: 45.27%, SPOT: 49.92%, DIS: 33.54%
            hist_vols = {
                'NFLX': 0.4527,
                'SPOT': 0.4992,
                'DIS': 0.3354
            }
            
            # Extract correlation matrix
            # From output correlation matrix
            self.correlation_matrix = np.array([
                [1.000, 0.357, 0.352],
                [0.357, 1.000, 0.489],
                [0.352, 0.489, 1.000]
            ])
            
            print("Good:  Historical data loaded successfully")
            print(f"   Spot prices: {self.spot_prices}")
            print(f"   Historical vols: {hist_vols}")
            
            # Store historical vols for blending
            self.historical_vols = hist_vols
            
        except Exception as e:
            print(f"!! Warning !!: Could not load historical data: {str(e)}")
            print("Using default values...")
            self._set_default_historical()
    
    def _load_option_data(self):
        """Load option-implied data from synthetic generators"""
        print("\n--- Loading Option-Implied Data ---")
        
        try:
            # Try to load Heston parameters first (more realistic)
            heston_params_path = f"{self.data_dir}/synthetic_option/heston/heston_parameters.pkl"
            bs_params_path = f"{self.data_dir}/synthetic_option/black_scholes/bs_surface_params.pkl"
            
            self.option_vols = {}
            
            # Load Heston parameters
            if os.path.exists(heston_params_path):
                with open(heston_params_path, 'rb') as f:
                    heston_params = pickle.load(f)
                
                print("Good:  Loaded Heston parameters")
                
                # Extract implied vols (using long-term vol as base)
                for ticker in self.tickers:
                    if ticker in heston_params:
                        # From Heston output:
                        # NFLX: theta=0.06 -> vol=24.5%
                        # SPOT: theta=0.05 -> vol=22.4%
                        # DIS: theta=0.04 -> vol=20.0%
                        theta = heston_params[ticker]['theta']
                        self.option_vols[ticker] = np.sqrt(theta)
                        
                        print(f"   {ticker}: Heston long-term vol = {self.option_vols[ticker]:.2%}")
            
            # Also load BS parameters for comparison
            if os.path.exists(bs_params_path):
                with open(bs_params_path, 'rb') as f:
                    bs_params = pickle.load(f)
                
                print("Good:  Loaded Black-Scholes parameters")
                
                # From BS output:
                # NFLX base_vol=0.35, SPOT base_vol=0.30, DIS base_vol=0.25
                for ticker in self.tickers:
                    if ticker in bs_params:
                        print(f"   {ticker}: BS base vol = {bs_params[ticker]['base_vol']:.2%}")
            
            # If no option vols loaded, use default from BS
            if not self.option_vols:
                self.option_vols = {
                    'NFLX': 0.35,
                    'SPOT': 0.30,
                    'DIS': 0.25
                }
                print("Using Black-Scholes base vols as option-implied")
            
        except Exception as e:
            print(f"!! Warning !!: Could not load option data: {str(e)}")
            self.option_vols = {
                'NFLX': 0.35,
                'SPOT': 0.30,
                'DIS': 0.25
            }
    
    def _blend_parameters(self):
        """Blend historical and option-implied parameters"""
        print("\n--- Blending Parameters ---")
        
        # Risk-free rate (from option data)
        risk_free_rate = 0.05
        
        # Blend weights: 40% historical, 60% option-implied
        # (Option-implied is forward-looking)
        hist_weight = 0.4
        option_weight = 0.6
        
        for ticker in self.tickers:
            # Blend volatilities
            hist_vol = self.historical_vols[ticker]
            option_vol = self.option_vols[ticker]
            
            blended_vol = hist_weight * hist_vol + option_weight * option_vol
            self.volatilities[ticker] = blended_vol
            
            # Set drift to risk-free rate (risk-neutral pricing)
            # Add small dividend yield adjustment if needed
            self.drift_rates[ticker] = risk_free_rate
            
            print(f"\n{ticker}:")
            print(f"   Historical vol: {hist_vol:.2%}")
            print(f"   Option-implied vol: {option_vol:.2%}")
            print(f"   Blended vol: {blended_vol:.2%}")
            print(f"   Risk-neutral drift: {risk_free_rate:.2%}")
    
    def _set_default_historical(self):
        """Set default historical parameters if loading fails"""
        self.spot_prices = {
            'NFLX': 850.0,
            'SPOT': 350.0,
            'DIS': 110.0
        }
        
        self.historical_vols = {
            'NFLX': 0.4527,
            'SPOT': 0.4992,
            'DIS': 0.3354
        }
        
        self.correlation_matrix = np.array([
            [1.000, 0.357, 0.352],
            [0.357, 1.000, 0.489],
            [0.352, 0.489, 1.000]
        ])
    
    def simulate_gbm_paths(self) -> np.ndarray:
        """
        Simulate correlated GBM paths for all assets
        Using antithetic variates for variance reduction
        """
        print("\n" + "="*60)
        print("SIMULATING GBM PATHS")
        print("="*60)
        print(f"Number of simulations: {self.n_simulations:,}")
        print(f"Number of time steps: {self.n_steps}")
        print(f"Time horizon: {self.T:.2f} years")
        print(f"Time step: {self.dt*365:.2f} days")
        
        # Use half simulations with antithetic variates
        n_half = self.n_simulations // 2
        paths = np.zeros((self.n_simulations, self.n_steps + 1, self.n_assets))
        
        # Set initial prices
        for i, ticker in enumerate(self.tickers):
            paths[:, 0, i] = self.spot_prices[ticker]
        
        # Generate correlated random numbers
        random_normals = np.random.normal(0, 1, 
                                         (n_half, self.n_steps, self.n_assets))
        
        # Apply Cholesky for correlation
        correlated_shocks = random_normals @ self.cholesky_matrix.T
        
        # Simulate with antithetic variates
        for sim in range(n_half):
            for step in range(self.n_steps):
                # Original path
                for i, ticker in enumerate(self.tickers):
                    drift = (self.drift_rates[ticker] - 0.5 * self.volatilities[ticker]**2) * self.dt
                    diffusion = self.volatilities[ticker] * np.sqrt(self.dt) * correlated_shocks[sim, step, i]
                    
                    paths[sim, step + 1, i] = paths[sim, step, i] * np.exp(drift + diffusion)
                
                # Antithetic path (negative shocks)
                for i, ticker in enumerate(self.tickers):
                    drift = (self.drift_rates[ticker] - 0.5 * self.volatilities[ticker]**2) * self.dt
                    diffusion = self.volatilities[ticker] * np.sqrt(self.dt) * (-correlated_shocks[sim, step, i])
                    
                    paths[sim + n_half, step + 1, i] = paths[sim + n_half, step, i] * np.exp(drift + diffusion)
        
        print(f"Good:  Generated {self.n_simulations:,} paths with antithetic variates")
        
        return paths
    

    ################################################################################
    ## Discrete Version 1 ##
    def detect_barrier_touch(self, paths: np.ndarray) -> np.ndarray:
        """
        Detect if barrier was touched during the simulation
        """
        # Calculate barrier levels
        barrier_levels = np.array([self.spot_prices[ticker] * self.barrier_level 
                                   for ticker in self.tickers])
        
        # Check if any asset price goes below barrier at any time
        min_prices = np.min(paths, axis=1)  # Min over time
        barrier_touched = np.any(min_prices <= barrier_levels, axis=1)
        
        # Calculate time of first touch for statistics
        touch_times = []
        for sim in range(self.n_simulations):
            if barrier_touched[sim]:
                for step in range(1, self.n_steps + 1):
                    if np.any(paths[sim, step, :] <= barrier_levels):
                        touch_times.append(step * self.dt)
                        break
        
        self.barrier_touch_times = touch_times
        
        return barrier_touched
    
    ################################################################################
    ## Continuous Version 1 ##
    def detect_barrier_touch_brownian_bridge(self, paths: np.ndarray) -> np.ndarray:
        """
        Detect barrier touch with Brownian Bridge correction for continuous monitoring
        """
        n_simulations = paths.shape[0]
        n_steps = paths.shape[1]
        
        # Calculate barrier levels
        barrier_levels = np.array([self.spot_prices[ticker] * self.barrier_level 
                                for ticker in self.tickers])
        
        # Initialize barrier touch array
        barrier_touched = np.zeros(n_simulations, dtype=bool)
        
        # Brownian Bridge correction
        for sim in range(n_simulations):
            for step in range(n_steps - 1):
                for asset in range(self.n_assets):
                    S_t = paths[sim, step, asset]
                    S_next = paths[sim, step + 1, asset]
                    B = barrier_levels[asset]
                    sigma = self.volatilities[self.tickers[asset]]
                    dt = self.dt
                    
                    # Only apply correction if both endpoints above barrier
                    if S_t > B and S_next > B:
                        # Brownian Bridge probability of crossing
                        log_ratio_t = np.log(S_t / B)
                        log_ratio_next = np.log(S_next / B)
                        
                        # Avoid division by zero
                        if sigma**2 * dt > 1e-10:
                            exponent = -2 * log_ratio_t * log_ratio_next / (sigma**2 * dt)
                            
                            # Probability of NOT crossing
                            p_no_cross = np.exp(exponent)
                            
                            # Monte Carlo decision: did we cross?
                            if np.random.random() > p_no_cross:
                                barrier_touched[sim] = True
                                break
                    
                    # Check discrete touch (standard method)
                    elif S_t <= B or S_next <= B:
                        barrier_touched[sim] = True
                        break
                
                if barrier_touched[sim]:
                    break
        
        return barrier_touched
    ################################################################################

    
    def calculate_payoff(self, paths: np.ndarray, barrier_touched: np.ndarray) -> np.ndarray:
        """
        Calculate payoff for each simulation based on BRC terms
        """
        n_simulations = paths.shape[0]
        payoffs = np.zeros(n_simulations)
        
        # Calculate coupon payments
        n_coupons = len(self.payment_dates)
        coupon_amount = self.principal * self.coupon_rate * 0.25  # Quarterly coupon
        
        # Risk-free rate for discounting (within simulation)
        r = 0.05
        
        for sim in range(n_simulations):
            # Add all coupon payments (discounted to valuation date)
            for i, payment_date in enumerate(self.payment_dates[:-1]):  # Exclude maturity
                t = (payment_date - self.valuation_date).days / 365.0
                payoffs[sim] += coupon_amount * np.exp(-r * t)
            
            # Calculate principal repayment at maturity
            if not barrier_touched[sim]:
                # Scenario 1: No barrier touch - full principal
                payoffs[sim] += self.principal * np.exp(-r * self.T)
            else:
                # Barrier touched - check final prices
                final_prices = paths[sim, -1, :]
                initial_prices = paths[sim, 0, :]
                
                # Check if all final prices > initial prices
                if np.all(final_prices > initial_prices):
                    # Scenario 2: Full principal
                    payoffs[sim] += self.principal * np.exp(-r * self.T)
                else:
                    # Scenario 3: Physical delivery of worst performer
                    # Calculate performance relative to initial
                    performance = final_prices / initial_prices
                    worst_idx = np.argmin(performance)
                    
                    # Principal converted to shares then valued at final price
                    n_shares = self.principal / initial_prices[worst_idx]
                    delivery_value = n_shares * final_prices[worst_idx]
                    payoffs[sim] += delivery_value * np.exp(-r * self.T)
        
        return payoffs
    
    def calculate_price(self, paths: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate BRC price using Monte Carlo simulation
        """
        print("\n" + "="*60)
        print("PRICING BARRIER REVERSE CONVERTIBLE")
        print("="*60)
        print(f"Principal: ${self.principal:.0f}")
        print(f"Coupon Rate: {self.coupon_rate:.2%} p.a.")
        print(f"Barrier Level: {self.barrier_level:.0%}")
        print(f"Time to Maturity: {self.T:.2f} years")
        
        # Simulate paths if not provided
        if paths is None:
            paths = self.simulate_gbm_paths()
        
        # Detect barrier touches
        barrier_touched = self.detect_barrier_touch(paths)
        
        # Calculate payoffs
        payoffs = self.calculate_payoff(paths, barrier_touched)
        
        # Calculate statistics
        mean_price = np.mean(payoffs)
        std_price = np.std(payoffs)
        se_price = std_price / np.sqrt(self.n_simulations)
        
        # Confidence intervals
        ci_lower = np.percentile(payoffs, 2.5)
        ci_upper = np.percentile(payoffs, 97.5)
        
        # Calculate probabilities
        barrier_prob = np.mean(barrier_touched)
        
        # For barrier touched cases
        barrier_indices = np.where(barrier_touched)[0]
        if len(barrier_indices) > 0:
            final_prices = paths[barrier_indices, -1, :]
            initial_prices = paths[barrier_indices, 0, :]
            
            all_above_initial = np.all(final_prices > initial_prices, axis=1)
            full_principal_prob = np.mean(all_above_initial) * barrier_prob
            physical_delivery_prob = barrier_prob - full_principal_prob
        else:
            full_principal_prob = 0
            physical_delivery_prob = 0
        
        # Store results
        results = {
            'price': mean_price,
            'std_error': se_price,
            'confidence_interval': (ci_lower, ci_upper),
            'barrier_touch_probability': barrier_prob,
            'scenario_probabilities': {
                'No Barrier Touch': 1 - barrier_prob,
                'Barrier Touch + Full Principal': full_principal_prob,
                'Barrier Touch + Physical Delivery': physical_delivery_prob
            },
            'expected_coupons': len(self.payment_dates[:-1]) * self.principal * self.coupon_rate * 0.25,
            'expected_principal': mean_price - len(self.payment_dates[:-1]) * self.principal * self.coupon_rate * 0.25,
            'n_simulations': self.n_simulations,
            'valuation_date': self.valuation_date,
            'maturity_date': self.maturity_date,
            'barrier_level': self.barrier_level
        }
        
        # Print results
        self._print_pricing_results(results)
        
        return results
    
    def _print_pricing_results(self, results: Dict):
        """Print pricing results in formatted way"""
        print("\n" + "="*60)
        print("PRICING RESULTS")
        print("="*60)
        print(f"\nBRC Price: ${results['price']:.4f} (per ${self.principal:.0f} principal)")
        print(f"Standard Error: ±${results['std_error']:.4f}")
        print(f"95% Confidence Interval: [${results['confidence_interval'][0]:.4f}, "
              f"${results['confidence_interval'][1]:.4f}]")
        
        print(f"\nBarrier Touch Probability: {results['barrier_touch_probability']:.2%}")
        
        print("\nScenario Probabilities:")
        for scenario, prob in results['scenario_probabilities'].items():
            print(f"   {scenario}: {prob:.2%}")
        
        print(f"\nExpected Coupons: ${results['expected_coupons']:.4f}")
        print(f"Expected Principal Repayment: ${results['expected_principal']:.4f}")
        
        # Calculate yield
        total_return = (results['price'] / self.principal - 1) / self.T
        print(f"\nAnnualized Yield: {total_return:.2%}")
    
    def analyze_risk_metrics(self, paths: np.ndarray):
        """
        Calculate additional risk metrics
        """
        print("\n" + "="*60)
        print("RISK METRICS ANALYSIS")
        print("="*60)
        
        # Calculate barrier touch times
        if hasattr(self, 'barrier_touch_times') and self.barrier_touch_times:
            touch_times = np.array(self.barrier_touch_times)
            print(f"\nBarrier Touch Time Statistics:")
            print(f"   Mean time to touch: {np.mean(touch_times):.2f} years")
            print(f"   Median time to touch: {np.median(touch_times):.2f} years")
            print(f"   Earliest touch: {np.min(touch_times):.2f} years")
            
            # Conditional probability of touch by time
            times = np.linspace(0.1, self.T, 10)
            cond_probs = []
            for t in times:
                prob = np.mean(np.array(touch_times) <= t)
                cond_probs.append(prob)
            
            print(f"\nConditional Touch Probabilities:")
            for t, p in zip(times, cond_probs):
                print(f"   By {t:.1f} years: {p:.2%}")
        
        # Calculate worst performer statistics
        final_prices = paths[:, -1, :]
        initial_prices = paths[:, 0, :]
        performances = final_prices / initial_prices
        
        worst_performers = np.argmin(performances, axis=1)
        print(f"\nWorst Performer Distribution:")
        for i, ticker in enumerate(self.tickers):
            prob = np.mean(worst_performers == i)
            print(f"   {ticker}: {prob:.2%}")
    
    def create_visualizations(self, paths: np.ndarray, results: Dict):
        """
        Create comprehensive visualizations
        """
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Sample price paths
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('GBM Simulation Results - Barrier Reverse Convertible', 
                    fontsize=16, fontweight='bold')
        
        # Plot sample paths for each asset
        n_sample = min(50, self.n_simulations)
        sample_idx = np.random.choice(self.n_simulations, n_sample, replace=False)
        time_axis = np.linspace(0, self.T, self.n_steps + 1)
        
        for idx, ticker in enumerate(self.tickers):
            ax = axes[idx // 2, idx % 2]
            
            for sim in sample_idx:
                ax.plot(time_axis, paths[sim, :, idx], alpha=0.2, linewidth=0.5)
            
            # Add barrier line
            barrier = self.spot_prices[ticker] * self.barrier_level
            ax.axhline(y=barrier, color='red', linestyle='--', 
                      linewidth=2, label=f'Barrier (${barrier:.1f})')
            
            ax.set_title(f'{ticker} - Sample Price Paths')
            ax.set_xlabel('Time (years)')
            ax.set_ylabel('Price ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/sample_paths.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Payoff distribution
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Calculate payoffs for distribution
        barrier_touched = self.detect_barrier_touch(paths)
        payoffs = self.calculate_payoff(paths, barrier_touched)
        
        # Histogram of payoffs
        axes[0].hist(payoffs, bins=50, edgecolor='black', alpha=0.7, density=True)
        axes[0].axvline(x=results['price'], color='red', linestyle='--', 
                        linewidth=2, label=f'Price: ${results["price"]:.2f}')
        axes[0].axvline(x=self.principal, color='green', linestyle=':', 
                        linewidth=2, label=f'Principal: ${self.principal:.0f}')
        axes[0].set_title('Distribution of Discounted Payoffs')
        axes[0].set_xlabel('Payoff Value ($)')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_payoffs = np.sort(payoffs)
        cumulative = np.arange(1, len(sorted_payoffs) + 1) / len(sorted_payoffs)
        axes[1].plot(sorted_payoffs, cumulative)
        axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        axes[1].axvline(x=self.principal, color='green', linestyle=':', 
                        label=f'Principal')
        axes[1].set_title('Cumulative Distribution Function')
        axes[1].set_xlabel('Payoff Value ($)')
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/payoff_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Barrier analysis
        if hasattr(self, 'barrier_touch_times') and self.barrier_touch_times:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            touch_times = np.array(self.barrier_touch_times)
            ax.hist(touch_times, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(x=np.mean(touch_times), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(touch_times):.2f}Y')
            ax.set_title('Distribution of Barrier Touch Times')
            ax.set_xlabel('Time to First Barrier Touch (years)')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/plots/barrier_times.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        print(f"Good:  Visualizations saved to {self.output_dir}/plots/")
    
    def save_results(self, paths: np.ndarray, results: Dict):
        """
        Save all results and parameters
        """
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        # Save model parameters
        params = {
            'tickers': self.tickers,
            'valuation_date': self.valuation_date,
            'maturity_date': self.maturity_date,
            'T': self.T,
            'barrier_level': self.barrier_level,
            'coupon_rate': self.coupon_rate,
            'principal': self.principal,
            'spot_prices': self.spot_prices,
            'drift_rates': self.drift_rates,
            'volatilities': self.volatilities,
            'correlation_matrix': self.correlation_matrix.tolist(),
            'n_simulations': self.n_simulations,
            'n_steps': self.n_steps,
            'payment_dates': [d.strftime('%Y-%m-%d') for d in self.payment_dates]
        }
        
        with open(f"{self.output_dir}/results/model_params.pkl", 'wb') as f:
            pickle.dump(params, f)
        
        # Save pricing results
        with open(f"{self.output_dir}/results/pricing_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Save summary statistics of paths
        path_stats = {
            'mean': np.mean(paths, axis=0),
            'std': np.std(paths, axis=0),
            'percentiles': {
                0.05: np.percentile(paths, 5, axis=0),
                0.25: np.percentile(paths, 25, axis=0),
                0.50: np.percentile(paths, 50, axis=0),
                0.75: np.percentile(paths, 75, axis=0),
                0.95: np.percentile(paths, 95, axis=0)
            }
        }
        
        with open(f"{self.output_dir}/simulations/path_statistics.pkl", 'wb') as f:
            pickle.dump(path_stats, f)
        
        print(f"Good:  All results saved to {self.output_dir}/")
        
        # Generate report
        self.generate_report(results, params)
    
    def generate_report(self, results: Dict, params: Dict):
        """
        Generate comprehensive pricing report
        """
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("BARRIER REVERSE CONVERTIBLE - COMPREHENSIVE PRICING REPORT")
        report_lines.append("="*70)
        report_lines.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report_lines.append("\n" + "-"*50)
        report_lines.append("PRODUCT SPECIFICATIONS")
        report_lines.append("-"*50)
        report_lines.append(f"Underlying Assets: {', '.join(self.tickers)}")
        report_lines.append(f"Valuation Date: {self.valuation_date.strftime('%Y-%m-%d')}")
        report_lines.append(f"Maturity Date: {self.maturity_date.strftime('%Y-%m-%d')}")
        report_lines.append(f"Time to Maturity: {self.T:.2f} years")
        report_lines.append(f"Coupon Rate: {self.coupon_rate:.2%} p.a.")
        report_lines.append(f"Barrier Level: {self.barrier_level:.0%} of Initial")
        report_lines.append(f"Principal Amount: ${self.principal:.0f}")
        
        report_lines.append("\n" + "-"*50)
        report_lines.append("CALIBRATED MODEL PARAMETERS")
        report_lines.append("-"*50)
        report_lines.append("\nSpot Prices:")
        for ticker, price in self.spot_prices.items():
            report_lines.append(f"   {ticker}: ${price:.2f}")
        
        report_lines.append("\nVolatilities (annualized):")
        for ticker, vol in self.volatilities.items():
            report_lines.append(f"   {ticker}: {vol:.2%}")
            report_lines.append(f"      Historical: {self.historical_vols[ticker]:.2%}")
            report_lines.append(f"      Option-Implied: {self.option_vols[ticker]:.2%}")
        
        report_lines.append("\nCorrelation Matrix:")
        corr_df = pd.DataFrame(self.correlation_matrix, 
                              index=self.tickers, columns=self.tickers)
        for line in corr_df.to_string().split('\n'):
            report_lines.append(f"   {line}")
        
        report_lines.append("\n" + "-"*50)
        report_lines.append("PRICING RESULTS")
        report_lines.append("-"*50)
        report_lines.append(f"\nBRC Price: ${results['price']:.4f}")
        report_lines.append(f"Standard Error: ±${results['std_error']:.4f}")
        report_lines.append(f"95% Confidence Interval: [${results['confidence_interval'][0]:.4f}, "
                           f"${results['confidence_interval'][1]:.4f}]")
        
        report_lines.append(f"\nBarrier Touch Probability: {results['barrier_touch_probability']:.2%}")
        
        report_lines.append("\nScenario Probabilities:")
        for scenario, prob in results['scenario_probabilities'].items():
            report_lines.append(f"   {scenario}: {prob:.2%}")
        
        report_lines.append(f"\nExpected Coupons: ${results['expected_coupons']:.4f}")
        report_lines.append(f"Expected Principal Repayment: ${results['expected_principal']:.4f}")
        
        annual_yield = (results['price'] / self.principal - 1) / self.T
        report_lines.append(f"\nAnnualized Yield: {annual_yield:.2%}")
        
        report_lines.append("\n" + "-"*50)
        report_lines.append("SIMULATION DETAILS")
        report_lines.append("-"*50)
        report_lines.append(f"Number of Simulations: {self.n_simulations:,}")
        report_lines.append(f"Number of Time Steps: {self.n_steps}")
        report_lines.append(f"Time Step Size: {self.dt:.4f} years (~{self.dt*365:.1f} days)")
        report_lines.append(f"Number of Coupon Payments: {len(self.payment_dates)}")
        
        report_lines.append("\n" + "-"*50)
        report_lines.append("COUPON PAYMENT DATES")
        report_lines.append("-"*50)
        for i, date in enumerate(self.payment_dates[:-1], 1):
            report_lines.append(f"   Payment {i}: {date.strftime('%Y-%m-%d')}")
        report_lines.append(f"   Maturity: {self.payment_dates[-1].strftime('%Y-%m-%d')}")
        
        report_lines.append("\n" + "="*70)
        
        # Write report to file
        with open(f"{self.output_dir}/results/pricing_report.txt", 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\nGood:  Report saved to {self.output_dir}/results/pricing_report.txt")
    
    def run_full_pricing(self) -> Dict:
        """
        Run complete pricing workflow
        """
        print("\n" + "="*70)
        print("GBM BARRIER REVERSE CONVERTIBLE PRICING - STARTED")
        print("="*70)
        
        # Step 1: Simulate paths
        paths = self.simulate_gbm_paths()
        
        # Step 2: Calculate price
        results = self.calculate_price(paths)
        
        # Step 3: Analyze risk metrics
        self.analyze_risk_metrics(paths)
        
        # Step 4: Create visualizations
        self.create_visualizations(paths, results)
        
        # Step 5: Save results
        self.save_results(paths, results)
        
        print("\n" + "="*70)
        print("PRICING COMPLETE")
        print("="*70)
        print(f"All outputs saved to: {self.output_dir}/")
        
        return results


def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("GBM MODEL FOR BARRIER REVERSE CONVERTIBLE")
    print("="*70)
    
    # Initialize GBM model with actual data
    brc_pricer = GBMBarrierReverseConvertible(
        tickers=['NFLX', 'SPOT', 'DIS'],
        valuation_date='2025-04-01',
        maturity_date='2026-10-01',
        barrier_level=0.50,
        coupon_rate=0.1375,
        principal=100.0,
        n_simulations=1000,
        n_steps=378,  # ~18 months of trading days
        data_dir='../../final_dataset'
    )
    
    # Run full pricing
    results = brc_pricer.run_full_pricing()
    
    print("\n" + "="*70)
    print("-- ! GBM Simulation Done ! --")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()


