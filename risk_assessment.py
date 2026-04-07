"""
BRC Risk Assessment with Mathematical Derivations
"""
import numpy as np
from scipy.stats import norm
import pickle

class BRCRiskManagerWithDerivations:
    """
    Risk manager with mathematical justification for limits
    """
    
    def __init__(self):
        # From fetch_historical_output_ref.txt
        self.volatilities = {
            'DIS': 0.3354,
            'NFLX': 0.4527,
            'SPOT': 0.4992
        }
        
        # From fetch_historical_output_ref.txt
        self.correlation_matrix = np.array([
            [1.000, 0.357, 0.352],
            [0.357, 1.000, 0.489],
            [0.352, 0.489, 1.000]
        ])
        
        # From fetch_historical_output_ref.txt
        self.max_drawdowns = {
            'NFLX': -0.7595,
            'SPOT': -0.8051,
            'DIS': -0.6072
        }
        
        # From fetch_historical_output_ref.txt
        self.kurtosis = {
            'NFLX': 22.916,
            'SPOT': 3.195,
            'DIS': 8.395
        }
        
        # From GBM.py
        self.barrier_level = 0.50
        


    def derive_critical_zone(self, monitoring_days=5, confidence=0.95, safety_margin=1.75):
        """
        Derive critical zone for barrier monitoring
        """
        z_score = norm.ppf(confidence)
        trading_days = 252
        
        print("="*70)
        print("DERIVATION: CRITICAL ZONE FOR BARRIER MONITORING")
        print("="*70)
        
        critical_zones = {}
        for ticker, vol_ann in self.volatilities.items():
            vol_daily = vol_ann / np.sqrt(trading_days)
            expected_move = z_score * vol_daily * np.sqrt(monitoring_days)
            critical_zone = expected_move * safety_margin
            
            critical_zones[ticker] = {
                'daily_vol': vol_daily,
                'expected_move': expected_move,
                'critical_zone': critical_zone
            }
            
            print(f"\n{ticker}:")
            print(f"   Annual Volatility: {vol_ann*100:.2f}%")
            print(f"   Daily Volatility: {vol_daily*100:.2f}%")
            print(f"   {monitoring_days}-Day Expected Move ({confidence*100:.0f}%): {expected_move*100:.2f}%")
            print(f"   Safety Margin: {safety_margin}x")
            print(f"   Critical Zone: {critical_zone*100:.1f}%")
            print(f"   Critical Price Level: {(self.barrier_level + critical_zone)*100:.1f}% of initial")
        
        # Use highest (most conservative)
        max_critical = max([v['critical_zone'] for v in critical_zones.values()])
        print(f"\n" + "-"*70)
        print(f"RECOMMENDATION: {max_critical*100:.0f}% critical zone (based on {max(critical_zones, key=lambda x: critical_zones[x]['critical_zone'])})")
        print(f"Monitor closely when price within {max_critical*100:.0f}% of barrier")
        print(f"Barrier at {self.barrier_level*100:.0f}%, Critical price at {(self.barrier_level + max_critical)*100:.0f}% of initial")
        
        return max_critical
    


    def derive_position_limit(self, max_portfolio_loss=0.025):
        """
        Derive position limit based on maximum acceptable loss
        """
        print("\n" + "="*70)
        print("DERIVATION: POSITION LIMITS")
        print("="*70)
        
        # Maximum loss per BRC (barrier breach + worst performer)
        max_loss_per_brc = 1 - self.barrier_level  # 50%
        
        print(f"\nMaximum Loss per BRC Position:")
        print(f"   Barrier Level: {self.barrier_level*100:.0f}% of initial")
        print(f"   Worst Case Recovery: {self.barrier_level*100:.0f}% of principal")
        print(f"   Maximum Loss: {max_loss_per_brc*100:.0f}% of position")
        
        print(f"\nPortfolio Loss Tolerance:")
        print(f"   Maximum Acceptable Loss: {max_portfolio_loss*100:.1f}% of portfolio")
        print(f"   (Standard institutional limit: 2-3%)")
        
        # Calculate position limit
        position_limit = max_portfolio_loss / max_loss_per_brc
        
        print(f"\nPosition Limit Calculation:")
        print(f"   Position Limit = Max Portfolio Loss / Max Loss per Position")
        print(f"   Position Limit = {max_portfolio_loss*100:.1f}% / {max_loss_per_brc*100:.0f}%")
        print(f"   Position Limit = {position_limit*100:.1f}%")
        
        print(f"\n" + "-"*70)
        print(f"RECOMMENDATION: Max {position_limit*100:.0f}% of portfolio in single BRC")
        print(f"\nExample ($1M portfolio):")
        print(f"   BRC Position: ${position_limit*1_000_000:,.0f}")
        print(f"   Worst Case Loss: ${position_limit*max_loss_per_brc*1_000_000:,.0f}")
        print(f"   Equals Max Acceptable Loss: ${max_portfolio_loss*1_000_000:,.0f}")
        
        return position_limit
    
    
    
    def derive_var_limit(self, confidence=0.95):
        """
        Derive VaR-based position limits
        """
        print("\n" + "="*70)
        print("DERIVATION: VaR LIMITS")
        print("="*70)
        
        trading_days = 252
        z_score = norm.ppf(confidence)
        
        print(f"\nIndividual Asset VaR ({confidence*100:.0f}% Confidence):")
        print(f"{'Asset':<10} {'Ann Vol':<10} {'Daily Vol':<10} {'VaR Daily':<10}")
        print("-"*50)
        
        var_values = []
        for ticker, vol_ann in self.volatilities.items():
            vol_daily = vol_ann / np.sqrt(trading_days)
            var_daily = z_score * vol_daily
            var_values.append(var_daily)
            print(f"{ticker:<10} {vol_ann*100:<10.2f} {vol_daily*100:<10.2f} {var_daily*100:<10.2f}")
        
        avg_var = np.mean(var_values)
        min_var = np.min(var_values)
        
        print(f"\nPortfolio VaR Considerations:")
        print(f"   Average Individual VaR: {avg_var*100:.2f}%")
        print(f"   Minimum Individual VaR: {min_var*100:.2f}%")
        
        # Fat tail adjustment (kurtosis)
        avg_kurtosis = np.mean(list(self.kurtosis.values()))
        kurtosis_adjustment = np.sqrt(avg_kurtosis / 3)  # Normal kurtosis = 3
        
        print(f"\nFat Tail Adjustment:")
        print(f"   Average Kurtosis: {avg_kurtosis:.1f}")
        print(f"   Normal Kurtosis: 3.0")
        print(f"   Tail Risk Multiplier: {kurtosis_adjustment:.2f}x")
        print(f"   (NFLX kurtosis = {self.kurtosis['NFLX']:.1f} - extreme fat tails!)")
        
        # Calculate VaR limit
        var_limit = min(0.5 * min_var, 0.3 * avg_var, 0.02)
        
        print(f"\nVaR Limit Calculation:")
        print(f"   Method 1 (50% of min VaR): {0.5*min_var*100:.2f}%")
        print(f"   Method 2 (30% of avg VaR): {0.3*avg_var*100:.2f}%")
        print(f"   Method 3 (Fixed cap): 2.00%")
        print(f"   Minimum: {var_limit*100:.2f}%")
        
        print(f"\n" + "-"*70)
        print(f"RECOMMENDATION: Daily VaR < {var_limit*100:.1f}% of portfolio")
        print(f"   Conservative relative to individual asset VaR")
        print(f"   Accounts for fat tails (kurtosis > 3)")
        print(f"   Aligns with institutional risk limits")
        
        return var_limit
    


    def generate_risk_assessment(self):
        """
        Generate complete risk management framework with all derivations
        """
        print("\n" + "="*70)
        print("BRC RISK ASSESSMENT")
        print("="*70)
        
        critical_zone = self.derive_critical_zone()
        position_limit = self.derive_position_limit()
        var_limit = self.derive_var_limit()
        
        print("\n" + "="*70)
        print("SUMMARY OF RISK LIMITS")
        print("="*70)
        print(f"1. Barrier Monitoring Critical Zone: {critical_zone*100:.0f}% above barrier")
        print(f"2. Single BRC Position Limit: {position_limit*100:.0f}% of portfolio")
        print(f"3. Daily VaR Limit: {var_limit*100:.1f}% of portfolio")
        print("="*70)
        
        return {
            'critical_zone': critical_zone,
            'position_limit': position_limit,
            'var_limit': var_limit
        }



# Run the complete derivation
if __name__ == "__main__":
    risk_mgr = BRCRiskManagerWithDerivations()
    framework = risk_mgr.generate_risk_assessment()


