import math
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class ShipDimensions:
    """
    Data class holding the main particulars of the ship.
    """
    L: float    # Length Between Perpendiculars (m)
    B: float    # Breadth (m)
    T: float    # Draft (m)
    Cb: float   # Block Coefficient
    Cp: float   # Prismatic Coefficient
    S: float    # Wetted Surface Area (m^2)
    
    # Optional parameters for appendage and transom (default to 0)
    Abt: float = 0.0 # Transverse area of bulbous bow (m^2)
    Cstern: float = 0.0 # Stern shape parameter (0 for normal, 10 for V-shape, -10 for U-shape)

    def __post_init__(self):
        """Validates that ship dimensions are physically possible."""
        if any(val <= 0 for val in [self.L, self.B, self.T, self.Cb, self.Cp, self.S]):
            raise ValueError("All main ship dimensions (L, B, T, Cb, Cp, S) must be strictly positive.")
        if self.Cb >= 1.0 or self.Cp >= 1.0:
            raise ValueError("Block and Prismatic coefficients must be less than 1.0.")

    @property
    def volume(self) -> float:
        """Calculates the displacement volume (m^3)."""
        return self.L * self.B * self.T * self.Cb

@dataclass
class WaterProperties:
    """
    Data class handling water density and kinematic viscosity.
    Defaults to Standard Salt Water at 15°C.
    """
    rho: float = 1025.0       # Density (kg/m^3)
    nu: float = 1.1892e-6     # Kinematic viscosity (m^2/s)

    @classmethod
    def fresh_water(cls) -> 'WaterProperties':
        """Returns standard fresh water properties at 15°C."""
        return cls(rho=999.0, nu=1.139e-6)
    
    @classmethod
    def salt_water(cls) -> 'WaterProperties':
        """Returns standard salt water properties at 15°C."""
        return cls(rho=1025.0, nu=1.1892e-6)

class HoltropCalculator:
    """
    Engine to calculate ship resistance components using the Holtrop & Mennen (1984) method.
    """
    G = 9.80665 # Acceleration due to gravity (m/s^2)

    def __init__(self, ship: ShipDimensions, water: WaterProperties):
        self.ship = ship
        self.water = water
        self._calculate_form_factor()

    def _calculate_form_factor(self):
        """
        Calculates the form factor (1 + k1) using Holtrop's 1982/1984 correlation.
        """
        # Length of run (approximate according to Holtrop)
        Lr = self.ship.L * (1 - self.ship.Cp + 0.06 * self.ship.Cp)
        
        # Stern shape coefficient
        c14 = 1.0 + 0.011 * self.ship.Cstern
        
        # Displacement volume
        vol = self.ship.volume
        
        # 1 + k1 calculation based on Holtrop regression
        term1 = (self.ship.B / self.ship.L) ** 1.06806
        term2 = (self.ship.T / self.ship.L) ** 0.46106
        term3 = (self.ship.L / Lr) ** 0.121563
        term4 = (self.ship.L ** 3 / vol) ** 0.36486
        term5 = (1 - self.ship.Cp) ** -0.604247
        
        self.k1 = 0.93 + 0.487118 * c14 * term1 * term2 * term3 * term4 * term5

    def get_froude_number(self, v_m_s: float) -> float:
        return v_m_s / math.sqrt(self.G * self.ship.L)

    def get_reynolds_number(self, v_m_s: float) -> float:
        return (v_m_s * self.ship.L) / self.water.nu

    def calculate_RF(self, v_m_s: float) -> Tuple[float, float]:
        """Calculates Frictional Resistance (RF) using ITTC-1957."""
        if v_m_s <= 0: return 0.0, 0.0
        Rn = self.get_reynolds_number(v_m_s)
        Cf = 0.075 / ((math.log10(Rn) - 2) ** 2)
        Rf = 0.5 * self.water.rho * (v_m_s ** 2) * self.ship.S * Cf
        return Rf, Cf

    def calculate_RW(self, v_m_s: float) -> float:
        """Calculates Wave-Making Resistance (RW) using Holtrop formulas."""
        if v_m_s <= 0: return 0.0
        Fn = self.get_froude_number(v_m_s)
        vol = self.ship.volume
        
        # Empirical estimate for Half-angle of entrance (iE) in degrees
        iE = 1.0 + 89.0 * math.exp(
            -(self.ship.L/self.ship.B)**0.80856 * 
            (1 - self.ship.Cw)**0.30484 if hasattr(self.ship, 'Cw') else 
            -(self.ship.L/self.ship.B)**0.8 * (1 - self.ship.Cb)**0.3 # Fallback if Cw is missing
        )
        iE = max(10, min(iE, 45)) # Bound the angle to reasonable limits

        # Holtrop c1 coefficient
        c1 = 2223105 * (self.ship.B / self.ship.L)**3.78613 * (self.ship.T / self.ship.B)**1.07961 * (90 - iE)**-1.37565
        
        # Holtrop m1 coefficient
        m1 = 0.01404 * (self.ship.L / self.ship.T) - 1.7525 * (vol**(1/3) / self.ship.L) \
             - 4.7932 * (self.ship.B / self.ship.L) - 8.0798 * self.ship.Cp \
             + 13.867 * (self.ship.Cp ** 2) - 6.984 * (self.ship.Cp ** 3)
        
        # Exponential term for low Froude numbers (Standard Holtrop approximation)
        d = -0.9
        try:
            Rw = self.water.rho * self.G * vol * c1 * math.exp(m1 * (Fn ** d))
        except OverflowError:
            Rw = 0.0 # Handles mathematically extreme limits at near-zero speeds
            
        return Rw

    def calculate_RA(self, v_m_s: float) -> float:
        """Calculates Model-Ship Correlation Allowance (RA)."""
        if v_m_s <= 0: return 0.0
        # Simplistic Holtrop correlation allowance coefficient Ca
        Ca = 0.006 * (self.ship.L + 100) ** -0.16 - 0.00205
        # Ensure Ca is not negative
        Ca = max(Ca, 0.0004) 
        Ra = 0.5 * self.water.rho * (v_m_s ** 2) * self.ship.S * Ca
        return Ra

    def calculate_total_resistance(self, speed_knots: float) -> dict:
        """
        Calculates all resistance components for a given speed in knots.
        Returns a dictionary of results.
        """
        v_m_s = speed_knots * 0.51444
        if v_m_s == 0:
            return {"Fn": 0, "Rn": 0, "RF": 0, "RW": 0, "RA": 0, "RT": 0, "PE": 0}

        Fn = self.get_froude_number(v_m_s)
        Rn = self.get_reynolds_number(v_m_s)
        
        Rf, _ = self.calculate_RF(v_m_s)
        Rw = self.calculate_RW(v_m_s)
        Ra = self.calculate_RA(v_m_s)
        
        # Optional appendages set to 0 as per instructions
        Rapp = 0.0 
        Rb = 0.0
        Rtr = 0.0
        
        Rt = Rf * self.k1 + Rw + Ra + Rapp + Rb + Rtr
        Pe = Rt * v_m_s # Effective Power (Watts)

        return {
            "Fn": Fn,
            "Rn": Rn,
            "RF": Rf / 1000, # Convert to kN
            "RW": Rw / 1000,
            "RA": Ra / 1000,
            "RT": Rt / 1000,
            "PE": Pe / 1000  # Convert to kW
        }

class ShipReport:
    """Handles formatting and outputting the resistance calculation results."""
    
    @staticmethod
    def generate_table(calculator: HoltropCalculator, min_speed: int, max_speed: int):
        print("\n" + "="*95)
        print(f"{'HOLTROP & MENNEN RESISTANCE CALCULATION REPORT':^95}")
        print("="*95)
        print(f"Ship: L={calculator.ship.L}m, B={calculator.ship.B}m, T={calculator.ship.T}m, "
              f"Cb={calculator.ship.Cb}, Cp={calculator.ship.Cp}")
        print(f"Water: Density={calculator.water.rho} kg/m^3, Kinematic Viscosity={calculator.water.nu} m^2/s")
        print(f"Calculated Form Factor (1 + k1): {calculator.k1:.4f}")
        print("-" * 95)
        
        # Table Header
        header = f"{'Speed(kts)':<12} | {'Fn':<8} | {'Rn':<12} | {'R_F (kN)':<10} | {'R_W (kN)':<10} | {'R_A (kN)':<10} | {'R_T (kN)':<10} | {'P_E (kW)':<10}"
        print(header)
        print("-" * 95)
        
        # Table Rows
        for v in range(min_speed, max_speed + 1):
            res = calculator.calculate_total_resistance(v)
            row = (f"{v:<12} | {res['Fn']:<8.3f} | {res['Rn']:<12.2e} | "
                   f"{res['RF']:<10.1f} | {res['RW']:<10.1f} | {res['RA']:<10.1f} | "
                   f"{res['RT']:<10.1f} | {res['PE']:<10.1f}")
            print(row)
            
        print("="*95)

def run_cli():
    """Command Line Interface to collect ship data or use defaults."""
    print("Welcome to the Holtrop-Mennen Ship Resistance Calculator!")
    print("Press [ENTER] at any prompt to use standard Merchant Vessel sample data.\n")

    try:
        raw_l = input("Enter Length (L) in meters [default 120]: ")
        if raw_l.strip() == "":
            print("=> Loading Default Sample Data for Merchant Vessel...")
            ship = ShipDimensions(L=120.0, B=20.0, T=7.0, Cb=0.75, Cp=0.76, S=3200.0)
            water = WaterProperties.salt_water()
            min_speed, max_speed = 1, 15
        else:
            l = float(raw_l)
            b = float(input("Enter Breadth (B) in meters: "))
            t = float(input("Enter Draft (T) in meters: "))
            cb = float(input("Enter Block Coefficient (Cb): "))
            cp = float(input("Enter Prismatic Coefficient (Cp): "))
            s = float(input("Enter Wetted Surface Area (S) in m^2: "))
            
            w_type = input("Water type - (S)alt or (F)resh? [default S]: ").upper()
            if w_type == 'F':
                water = WaterProperties.fresh_water()
            else:
                water = WaterProperties.salt_water()

            min_speed = int(input("Enter minimum speed (knots) [default 1]: ") or 1)
            max_speed = int(input("Enter maximum speed (knots) [default 20]: ") or 20)

            ship = ShipDimensions(L=l, B=b, T=t, Cb=cb, Cp=cp, S=s)

        # Initialize engine and generate report
        engine = HoltropCalculator(ship, water)
        ShipReport.generate_table(engine, min_speed, max_speed)

    except ValueError as e:
        print(f"\n[ERROR] Invalid Input: {e}")
        print("Please ensure numerical inputs are correct and positive. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    # Runs the out-of-the-box CLI interface.
    # To bypass CLI and run programmatically, you can instantiate the classes directly.
    run_cli()