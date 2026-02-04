"""
Management command to generate massive, realistic industrial data for RAG.

Generates:
- Equipment: ~500 pieces (already exists from populate_industrial_db)
- Alerts: 500+ realistic alerts across ALL equipment (not just 15 templates)
- Maintenance: 2000+ records across ALL equipment (not just 13 items)
- Operational Documents: 1500+ (SOPs, inspection reports, incident reports,
  energy audits, shift handover notes, commissioning reports, calibration certs)
- Energy Time-Series: 5000+ hourly/daily readings for trending
- Shift Logs: 1000+ shift handover entries
- Work Orders: 800+ open/closed work orders

All data is realistic for an industrial facility with transformers, DG sets,
HVAC, pumps, compressors, motors, and energy metering.

Usage:
    python manage.py generate_rich_data
    python manage.py generate_rich_data --clear      # Clear + regenerate
    python manage.py generate_rich_data --stats      # Show counts only
    python manage.py generate_rich_data --skip-equipment  # Skip equipment (use existing)
"""

import random
import uuid
import json
import math
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db import connection

from industrial.models import (
    Transformer, DieselGenerator, ElectricalPanel, UPS,
    Chiller, AHU, CoolingTower, Pump, Compressor, Motor,
    EnergyMeter, Alert, MaintenanceRecord,
)

# ============================================================
# Constants
# ============================================================

TECHNICIANS = [
    "Rajesh Kumar", "Sunil Sharma", "Manoj Singh", "Vikram Patel",
    "Arun Verma", "Deepak Mishra", "Sanjay Gupta", "Prakash Yadav",
    "Ramesh Joshi", "Abhishek Nair", "Kiran Reddy", "Ganesh Iyer",
    "Harish Menon", "Pradeep Das", "Santosh Pillai", "Nikhil Mehta",
    "Ravi Shankar", "Amit Tiwari", "Suresh Babu", "Vijay Rao",
]

VENDORS = [
    "ABB India Service", "Siemens Technical Support", "Schneider Electric Services",
    "L&T Electrical Maintenance", "Thermax Service", "Blue Star AMC",
    "Carrier India", "Trane Technologies", "Grundfos Service Centre",
    "Atlas Copco Service", "Kirloskar Brothers", "Crompton Greaves Services",
    "Emerson Network Power", "Eaton Service India", "Johnson Controls",
    "Honeywell Building Solutions", "Daikin India Service", "Voltas Service",
    "KSB Pumps Service", "Wilo India", "Internal Maintenance Team",
]

BUILDINGS = [
    "Main Plant", "Block A", "Block B", "Block C", "Utility Building",
    "Admin Building", "Warehouse", "Data Center", "R&D Lab",
    "Quality Lab", "Workshop", "Substation 1", "Substation 2",
]

SHIFTS = ["Morning (06:00-14:00)", "Afternoon (14:00-22:00)", "Night (22:00-06:00)"]

SHIFT_SUPERVISORS = [
    "S. Raghavan", "K. Narayanan", "P. Murugan", "R. Krishnan",
    "V. Subramanian", "A. Chandrasekhar", "M. Venkatesh", "D. Ramachandran",
]


def _r(low, high, decimals=1):
    """Random float in range, rounded."""
    return round(random.uniform(low, high), decimals)


def _ts(days_ago_max=365):
    """Random timestamp within last N days."""
    return timezone.now() - timedelta(
        days=random.randint(0, days_ago_max),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )


# ============================================================
# Alert Generation — realistic industrial alerts
# ============================================================

ALERT_TEMPLATES_BY_TYPE = {
    "transformer": [
        ("threshold", "high", "Oil temperature exceeded threshold", "oil_temperature", (75, 95), 75, "°C"),
        ("threshold", "high", "Winding temperature high", "winding_temperature", (80, 105), 85, "°C"),
        ("threshold", "medium", "Load exceeding 80% of rated capacity", "load_percent", (80, 98), 80, "%"),
        ("threshold", "critical", "Oil temperature critically high — risk of insulation degradation", "oil_temperature", (95, 115), 90, "°C"),
        ("threshold", "medium", "Output voltage deviation beyond ±5%", "output_voltage", (408, 460), None, "V"),
        ("threshold", "low", "Impedance test deviation detected", "impedance_percent", (5.5, 8.0), 6.0, "%"),
        ("fault", "critical", "Buchholz relay tripped — possible internal fault", None, None, None, None),
        ("fault", "high", "Oil level low in conservator tank", None, None, None, None),
        ("fault", "critical", "Differential protection operated — transformer isolated", None, None, None, None),
        ("maintenance", "medium", "Scheduled oil filtration overdue by 30 days", None, None, None, None),
        ("maintenance", "low", "Silica gel breather needs replacement — saturated", None, None, None, None),
        ("safety", "critical", "PRV (Pressure Relief Valve) operated — check immediately", None, None, None, None),
    ],
    "diesel_generator": [
        ("threshold", "high", "Coolant temperature high", "coolant_temperature", (90, 110), 90, "°C"),
        ("threshold", "critical", "Oil pressure low", "oil_pressure", (1.5, 2.8), 2.5, "bar"),
        ("threshold", "medium", "Fuel level below 30%", "fuel_level_percent", (10, 29), 30, "%"),
        ("threshold", "high", "Battery voltage low — cranking may fail", "battery_voltage", (20, 23.5), 24, "V"),
        ("threshold", "medium", "Engine vibration above normal", "vibration", (6.0, 12.0), 5.0, "mm/s"),
        ("fault", "critical", "Engine failed to start on auto command", None, None, None, None),
        ("fault", "critical", "Overspeed trip — engine shut down", None, None, None, None),
        ("fault", "high", "AMF panel communication failure", None, None, None, None),
        ("maintenance", "medium", "Engine oil change overdue — 500+ hours past interval", None, None, None, None),
        ("maintenance", "low", "Air filter replacement due", None, None, None, None),
        ("safety", "critical", "Exhaust temperature abnormally high — check turbocharger", None, None, None, None),
    ],
    "electrical_panel": [
        ("threshold", "high", "Bus bar temperature high — thermal imaging detected hotspot", "temperature", (65, 90), 60, "°C"),
        ("threshold", "medium", "Phase imbalance — current asymmetry > 10%", "current_imbalance", (10, 25), 10, "%"),
        ("threshold", "medium", "Power factor below acceptable limit", "power_factor", (0.72, 0.84), 0.85, None),
        ("threshold", "high", "Neutral current abnormally high — possible harmonics", "neutral_current", (50, 120), 40, "A"),
        ("fault", "critical", "Earth fault detected on outgoing feeder", None, None, None, None),
        ("fault", "high", "MCCB tripped on overcurrent", None, None, None, None),
        ("fault", "critical", "Arc flash detected — panel isolated", None, None, None, None),
        ("communication", "medium", "SCADA communication lost for > 15 minutes", None, None, None, None),
        ("maintenance", "low", "Capacitor bank inspection due", None, None, None, None),
        ("maintenance", "medium", "Thermal imaging inspection overdue", None, None, None, None),
    ],
    "ups": [
        ("threshold", "critical", "Battery backup below 50% — critical load at risk", "battery_percent", (25, 49), 50, "%"),
        ("threshold", "high", "Output voltage deviation > 3%", "output_voltage_actual", (210, 222), None, "V"),
        ("threshold", "medium", "Load exceeding 75% of capacity", "load_percent", (75, 95), 75, "%"),
        ("threshold", "high", "Battery temperature high", "temperature", (38, 50), 35, "°C"),
        ("fault", "critical", "UPS switched to bypass mode — load unprotected", None, None, None, None),
        ("fault", "critical", "Inverter fault — immediate attention required", None, None, None, None),
        ("fault", "high", "Rectifier/charger failure", None, None, None, None),
        ("maintenance", "medium", "Battery impedance test overdue", None, None, None, None),
        ("maintenance", "high", "Battery end-of-life warning — replace within 3 months", None, None, None, None),
        ("communication", "medium", "BMS communication lost", None, None, None, None),
    ],
    "chiller": [
        ("threshold", "high", "Compressor discharge pressure high", "condenser_pressure", (15, 20), 15, "bar"),
        ("threshold", "medium", "Chilled water supply temperature above setpoint", "chilled_water_supply_temp", (8.5, 12), 8, "°C"),
        ("threshold", "high", "Compressor current above rated limit", "compressor_current", (280, 350), 270, "A"),
        ("threshold", "medium", "Condenser approach temperature high — fouling suspected", "approach_temp", (5, 10), 4, "°C"),
        ("threshold", "low", "Evaporator pressure low — possible refrigerant leak", "evaporator_pressure", (2.0, 2.8), 3.0, "bar"),
        ("fault", "critical", "Compressor tripped on high discharge temperature", None, None, None, None),
        ("fault", "high", "Low refrigerant charge detected", None, None, None, None),
        ("fault", "critical", "Oil pressure differential switch tripped", None, None, None, None),
        ("maintenance", "medium", "Condenser tube cleaning due — efficiency dropping", None, None, None, None),
        ("maintenance", "low", "Refrigerant leak test due", None, None, None, None),
    ],
    "ahu": [
        ("threshold", "medium", "Filter differential pressure high — filter clogged", "filter_dp", (250, 400), 250, "Pa"),
        ("threshold", "medium", "Supply air temperature above setpoint", "supply_air_temp", (18, 24), 16, "°C"),
        ("threshold", "low", "Return air humidity high", "return_air_humidity", (70, 85), 65, "%RH"),
        ("threshold", "high", "Fan motor current above rated — possible belt issue", "motor_current", (25, 40), 22, "A"),
        ("fault", "high", "VFD fault — fan running at fixed speed", None, None, None, None),
        ("fault", "medium", "Chilled water valve actuator failure — stuck at position", None, None, None, None),
        ("fault", "high", "Belt snapped — fan not running", None, None, None, None),
        ("communication", "medium", "DDC controller offline", None, None, None, None),
        ("maintenance", "medium", "Filter replacement due", None, None, None, None),
        ("maintenance", "low", "Fan belt inspection due", None, None, None, None),
    ],
    "cooling_tower": [
        ("threshold", "medium", "Water conductivity high — blowdown needed", "conductivity", (1500, 2500), 1500, "µS/cm"),
        ("threshold", "high", "Outlet water temperature high — cooling degraded", "outlet_water_temp", (33, 40), 32, "°C"),
        ("threshold", "medium", "Water level low in basin", "water_level_percent", (40, 59), 60, "%"),
        ("threshold", "low", "Fan vibration elevated — check alignment", "vibration", (4.0, 8.0), 3.5, "mm/s"),
        ("fault", "high", "Fan motor overload tripped", None, None, None, None),
        ("fault", "medium", "Float valve malfunction — overflow detected", None, None, None, None),
        ("maintenance", "medium", "Chemical dosing system check due", None, None, None, None),
        ("maintenance", "low", "Fill media inspection due", None, None, None, None),
    ],
    "pump": [
        ("threshold", "high", "Bearing temperature high", "bearing_temperature", (65, 85), 65, "°C"),
        ("threshold", "medium", "Vibration elevated — check alignment and impeller", "vibration", (4.5, 8.0), 4.5, "mm/s"),
        ("threshold", "medium", "Discharge pressure low — possible cavitation", "discharge_pressure", (1.0, 2.5), 2.5, "bar"),
        ("threshold", "high", "Motor current above rated — check mechanical seal", "motor_current", (30, 50), 28, "A"),
        ("threshold", "low", "Flow rate below design — check strainer", "flow_rate_actual", (15, 40), 50, "m³/hr"),
        ("fault", "critical", "Seal failure — water leakage detected", None, None, None, None),
        ("fault", "high", "Motor winding temperature trip", None, None, None, None),
        ("fault", "medium", "VFD fault on pump motor", None, None, None, None),
        ("maintenance", "medium", "Mechanical seal replacement due", None, None, None, None),
        ("maintenance", "low", "Coupling alignment check due", None, None, None, None),
    ],
    "compressor": [
        ("threshold", "high", "Discharge temperature high", "discharge_temperature", (90, 115), 90, "°C"),
        ("threshold", "medium", "Oil temperature above normal", "oil_temperature", (70, 90), 70, "°C"),
        ("threshold", "high", "Discharge pressure above setpoint", "discharge_pressure", (10, 14), 10, "bar"),
        ("threshold", "low", "Oil level low in separator", "oil_level", (30, 49), 50, "%"),
        ("fault", "critical", "Motor overload tripped — compressor shut down", None, None, None, None),
        ("fault", "high", "Air intake filter blocked — pressure drop > 500 mbar", None, None, None, None),
        ("fault", "medium", "Condensate drain valve stuck open", None, None, None, None),
        ("maintenance", "medium", "Oil separator element replacement due at 4000 hours", None, None, None, None),
        ("maintenance", "low", "Intake filter replacement due", None, None, None, None),
    ],
    "motor": [
        ("threshold", "high", "Winding temperature high", "winding_temperature", (80, 110), 80, "°C"),
        ("threshold", "high", "Bearing temperature high", "bearing_temperature", (65, 85), 65, "°C"),
        ("threshold", "medium", "Vibration elevated", "vibration", (4.5, 10.0), 4.5, "mm/s"),
        ("threshold", "medium", "Current imbalance between phases", "current_imbalance", (8, 20), 8, "%"),
        ("threshold", "low", "Power factor lower than expected", "power_factor", (0.65, 0.74), 0.75, None),
        ("fault", "critical", "Motor tripped on overcurrent protection", None, None, None, None),
        ("fault", "high", "Insulation resistance below minimum", None, None, None, None),
        ("fault", "medium", "VFD fault — motor running direct-on-line", None, None, None, None),
        ("maintenance", "medium", "Bearing lubrication due", None, None, None, None),
        ("maintenance", "low", "Vibration analysis due", None, None, None, None),
    ],
    "energy_meter": [
        ("threshold", "medium", "Maximum demand approaching contract limit", "max_demand_kw", (2800, 3200), 3000, "kW"),
        ("threshold", "high", "Power factor penalty zone — below 0.9", "power_factor", (0.82, 0.89), 0.9, None),
        ("threshold", "low", "Voltage THD above 5%", "voltage_thd", (5.0, 12.0), 5.0, "%"),
        ("threshold", "medium", "Phase voltage deviation > 3%", "voltage_deviation", (3.0, 8.0), 3.0, "%"),
        ("communication", "high", "Meter communication lost — billing data gap", None, None, None, None),
        ("fault", "medium", "CT polarity reversal detected", None, None, None, None),
        ("maintenance", "low", "Meter calibration due", None, None, None, None),
    ],
}

# ============================================================
# Maintenance work descriptions — domain-specific
# ============================================================

MAINT_WORK_BY_TYPE = {
    "transformer": {
        "preventive": [
            ("Annual transformer maintenance", "Checked oil level, silica gel breather, gaskets, bushings. Cleaned radiator fins. Tightened all terminal connections. Measured insulation resistance (IR values: HV-E: 2.5 GΩ, LV-E: 1.8 GΩ, HV-LV: 3.2 GΩ). All values within limits.", "Silica gel (5 kg), gasket set"),
            ("Transformer oil filtration", "Performed hot oil circulation filtration for 48 hours. Pre-filtration BDV: 42 kV. Post-filtration BDV: 72 kV. Moisture content reduced from 35 ppm to 12 ppm. Oil dielectric strength restored.", "Filter cartridges (4 nos)"),
            ("Bushing cleaning and inspection", "Cleaned all HV and LV bushings with isopropyl alcohol. Checked for cracks, carbon tracking. Measured bushing tan-delta. All bushings in good condition. Applied RTV silicone coating.", "RTV silicone, cleaning solvent"),
        ],
        "corrective": [
            ("Oil leak repair at drain valve", "Found oil seepage at bottom drain valve. Replaced gasket and tightened valve body. Topped up 50 liters of transformer oil. Monitored for 24 hours — no further leakage.", "Drain valve gasket, transformer oil (50L)"),
            ("Winding temperature indicator replacement", "WTI showing erratic readings. Replaced temperature sensing bulb and capillary tube. Calibrated new sensor against reference thermometer. Set alarm at 85°C, trip at 95°C.", "WTI assembly, capillary tube"),
            ("Cooling fan motor replacement", "Fan motor #2 seized. Removed and replaced with identical spare. Checked fan rotation direction and airflow. Tested auto-start on temperature rise.", "Fan motor 0.75kW, fan blades"),
        ],
        "breakdown": [
            ("Emergency: Buchholz relay trip investigation", "Buchholz relay tripped at 03:45. Gas analysis indicated dissolved gas — primarily hydrogen and acetylene. DGA results suggest possible partial discharge or arcing. Transformer de-energized. Gas relay reset after venting. Oil sample sent for complete DGA. Recommended frequency monitoring.", "Oil sample containers"),
        ],
        "inspection": [
            ("Quarterly visual inspection", "Inspected transformer externals: oil level normal, silica gel half saturated (blue), no visible leaks, bushings clean, OLTC counter reading: 14,523. Radiator fans operational. Ground connection secure. IR measurement: satisfactory.", ""),
            ("Thermographic survey", "Performed IR thermography on all connections. Hot spot found at LV bushing clamp — 72°C vs ambient 34°C. Recommended re-torque during next maintenance window. All other connections within 10°C differential.", ""),
        ],
    },
    "diesel_generator": {
        "preventive": [
            ("250-hour service", "Changed engine oil (Mobil Delvac 15W-40, 45L). Replaced oil filter, fuel filter (primary and secondary), and air filter. Checked coolant level and antifreeze concentration. Inspected belt tension and fan condition. Run test for 1 hour at 75% load.", "Engine oil 45L, oil filter, fuel filter x2, air filter"),
            ("Annual DG comprehensive service", "Full service including injector testing, valve clearance adjustment (intake: 0.25mm, exhaust: 0.45mm), turbocharger inspection, governor calibration, alternator winding IR test (all phases >100 MΩ). Battery load test — 8.2V under load (acceptable). AMF panel functional test passed.", "Injector nozzle set, valve cover gasket, V-belts x2, battery electrolyte"),
            ("Fuel system maintenance", "Drained fuel tank sediment (15L contaminated fuel removed). Cleaned fuel tank strainer. Replaced fuel return line. Bled air from fuel system. Test run confirmed clean fuel flow.", "Fuel return line, strainer element"),
        ],
        "corrective": [
            ("Starter motor repair", "DG failed to crank. Diagnosed faulty starter motor solenoid. Replaced solenoid and starter motor brushes. Cleaned battery terminals and applied anti-corrosion grease. Cranking restored — engine starts within 3 seconds.", "Starter solenoid, brush set, anti-corrosion grease"),
            ("Coolant leak repair", "Found coolant leak at radiator hose junction. Replaced upper radiator hose and clamps. Topped up coolant with 50/50 mix. Pressure tested cooling system at 1.2 bar — held for 30 minutes.", "Radiator hose, hose clamps x4, coolant 20L"),
        ],
        "breakdown": [
            ("Emergency: DG overspeed trip", "DG tripped on overspeed at 1620 RPM (limit: 1575 RPM) during load rejection. Governor linkage found loose. Re-adjusted governor and tightened all linkages. Speed sensing pickup gap adjusted to 0.5mm. Load rejection test passed — speed stayed within 1500±5%.", "Governor linkage pins"),
        ],
        "inspection": [
            ("Monthly DG readiness test", "Started DG on manual, checked all parameters: RPM 1500, voltage 415V (L-L), frequency 50.1 Hz, oil pressure 4.2 bar, coolant temp 78°C after 10 min. AMF auto-start test: engine cranked and synchronized in 12 seconds. Load bank test at 50%, 75%, 100% — all passed.", ""),
        ],
    },
    "chiller": {
        "preventive": [
            ("Annual chiller maintenance", "Inspected compressor oil level and condition. Checked refrigerant charge (R134a, 85 kg — within limits). Cleaned condenser tubes with mechanical brush — approach temperature improved from 5.2°C to 3.1°C. Checked expansion valve operation. Inspected electrical connections and control wiring. COP improved from 4.8 to 5.6.", "Compressor oil (10L), condenser cleaning brushes, gasket set"),
            ("Condenser tube cleaning", "Performed mechanical brush cleaning of condenser tubes (120 nos). Pre-cleaning: approach temp 6.8°C, condenser pressure 13.2 bar. Post-cleaning: approach temp 3.2°C, condenser pressure 11.5 bar. Significant efficiency improvement. Chemical treatment applied to prevent scale.", "Tube cleaning brushes, chemical treatment solution"),
            ("Evaporator inspection", "Inspected evaporator shell and tubes. No fouling detected. Checked evaporator pressure (4.2 bar — normal). Verified chilled water flow rate with ultrasonic meter — 180 m³/hr (design: 200 m³/hr). Recommended balancing valve adjustment.", ""),
        ],
        "corrective": [
            ("Refrigerant leak repair", "Leak detected at flare connection on liquid line. Isolated section, recovered refrigerant. Replaced flare nut and re-made connection. Vacuum held for 4 hours. Recharged with 12 kg R134a. System operational.", "R134a refrigerant (12 kg), flare nut"),
            ("VFD repair on condenser pump", "Condenser pump VFD showing E-015 (output phase loss). Found blown IGBT module. Replaced VFD board. Programmed motor parameters. Auto-tune completed. Pump running normally on VFD.", "VFD control board, IGBT module"),
        ],
        "inspection": [
            ("Quarterly chiller performance test", "Measured: CHWST 6.8°C, CHWRT 12.4°C, ΔT 5.6°C, flow 195 m³/hr, power 210 kW, capacity 1284 kW (365 TR), COP 6.1. All parameters within design. Next condenser cleaning recommended in 2 months.", ""),
        ],
    },
    "pump": {
        "preventive": [
            ("Annual pump overhaul", "Removed pump from base. Replaced mechanical seal (John Crane Type 21). Checked impeller for erosion — minor pitting, within limits. Replaced wear rings. Checked shaft runout — 0.03mm (limit: 0.05mm). Re-aligned pump-motor coupling. Vibration post-overhaul: 1.2 mm/s (pre: 3.8 mm/s).", "Mechanical seal, wear rings, coupling spider"),
            ("Bearing replacement", "Replaced both DE and NDE bearings (SKF 6310-2RS). Packed with Mobil Grease XHP 222. Measured shaft journal — within tolerance. Post-replacement bearing temperature: 42°C (was 68°C). Vibration: 0.8 mm/s.", "SKF 6310-2RS x2, bearing grease"),
            ("Quarterly pump maintenance", "Checked gland packing, adjusted to 2 drops/sec leakage. Greased bearings. Checked coupling alignment — angular: 0.05mm, parallel: 0.03mm. Cleaned strainer basket. Inspected base bolts.", "Gland packing (1 set)"),
        ],
        "corrective": [
            ("Mechanical seal replacement — leaking", "Pump leaking from mechanical seal. Shut down and isolated. Found carbon face worn and O-ring hardened. Replaced complete seal assembly. Flush plan 11 piping cleaned. Pump restarted — no leakage after 24 hours.", "Mechanical seal assembly, O-ring kit"),
        ],
        "inspection": [
            ("Vibration analysis", "Performed tri-axial vibration measurement on pump and motor. Pump DE horizontal: 2.1 mm/s, vertical: 1.8 mm/s, axial: 1.2 mm/s. Motor DE horizontal: 1.5 mm/s. All within ISO 10816 Zone A/B. No spectrum anomalies. Next measurement in 3 months.", ""),
        ],
    },
    "motor": {
        "preventive": [
            ("Annual motor maintenance", "Cleaned motor exterior and cooling fins. Checked terminal box connections — re-torqued to spec. Insulation resistance: U-E: 850 MΩ, V-E: 920 MΩ, W-E: 780 MΩ. Bearing condition good (vibration: 1.4 mm/s). Greased DE and NDE bearings per schedule.", "Bearing grease, cleaning solvent"),
            ("Motor winding varnishing", "Motor pulled for rewinding. Stator winding insulation test failed (IR < 5 MΩ). Re-varnished windings with Class F varnish. Baked at 150°C for 8 hours. Post-treatment IR: 1200 MΩ. Motor reinstalled and aligned.", "Winding varnish, insulation tape"),
        ],
        "corrective": [
            ("Bearing replacement — high vibration", "Motor vibration at 6.2 mm/s (alarm: 4.5). Spectrum analysis showed bearing defect frequency. Replaced DE bearing (SKF 6312). NDE bearing inspected — acceptable. Post-replacement vibration: 1.1 mm/s.", "SKF 6312, bearing grease"),
        ],
        "inspection": [
            ("Thermographic survey", "IR thermography on motor and connections. Motor frame: 62°C, terminal box: 58°C, cable lug: 55°C. All within limits. No hot spots detected. Motor efficiency appears normal.", ""),
            ("Insulation resistance test", "Measured IR with 1000V megger. U-E: 1.2 GΩ, V-E: 980 MΩ, W-E: 1.1 GΩ. Polarization index: 3.8 (good). Winding insulation in healthy condition. Next test due in 6 months.", ""),
        ],
    },
    "compressor": {
        "preventive": [
            ("4000-hour compressor service", "Replaced oil separator element, air intake filter, oil filter. Drained and refilled compressor oil (Roto-Inject Fluid, 28L). Checked minimum pressure valve. Cleaned oil cooler and aftercooler. Verified unloading valve operation. Specific energy: 5.8 kW/m³/min (design: 5.5).", "Oil separator element, air filter, oil filter, compressor oil 28L"),
            ("Quarterly compressor maintenance", "Inspected belt tension and condition. Checked condensate drain — auto-drain functional. Cleaned intake filter housing. Verified safety valve operation. Logged operating parameters: discharge pressure 7.5 bar, temperature 78°C, oil pressure 3.2 bar.", ""),
        ],
        "corrective": [
            ("Minimum pressure valve replacement", "Compressor unable to build pressure above 5 bar. Found minimum pressure valve stuck partially open. Replaced valve assembly. System pressure now reaches 7.5 bar setpoint. Checked for downstream leaks — none found.", "Minimum pressure valve assembly"),
        ],
        "inspection": [
            ("Air leak audit", "Performed compressed air leak audit using ultrasonic detector. Found 14 leaks totaling estimated 45 CFM. Major leaks: coupling disconnects (3), worn hose connections (4), pneumatic cylinder seals (3), FRL drains (4). Estimated annual savings if fixed: ₹3.2 lakhs.", ""),
        ],
    },
}

# Fallback for equipment types not explicitly listed
MAINT_WORK_GENERIC = {
    "preventive": [
        ("Scheduled preventive maintenance", "Completed scheduled PM activities per checklist. Inspected all mechanical and electrical components. Tightened connections. Cleaned equipment. Checked operational parameters — all within limits.", "Consumables as per checklist"),
    ],
    "corrective": [
        ("Corrective repair", "Diagnosed and repaired reported fault. Equipment tested and restored to normal operation.", "Replacement parts as needed"),
    ],
    "breakdown": [
        ("Emergency breakdown repair", "Equipment tripped unexpectedly. Root cause identified and rectified. System tested under load before handover to operations.", "Emergency spares"),
    ],
    "inspection": [
        ("Routine inspection", "Visual inspection completed. No anomalies found. Equipment condition satisfactory. All parameters within normal range.", ""),
    ],
}


# ============================================================
# Operational Documents
# ============================================================

SOP_TEMPLATES = [
    ("SOP-EL-001", "Transformer Energization Procedure", "transformer",
     "Standard Operating Procedure for energizing distribution and power transformers after maintenance or shutdown. "
     "Pre-energization checks: 1) Verify all maintenance work completed and clearances removed. "
     "2) Check oil level in conservator tank — must be between min and max marks. "
     "3) Verify silica gel breather color — blue indicates dry (acceptable). "
     "4) Check all bushing connections torqued to specification. "
     "5) Verify earth connections intact. "
     "6) Measure winding insulation resistance — must exceed 100 MΩ. "
     "7) Check Buchholz relay — both floats in normal position. "
     "8) Verify OLTC in neutral position. "
     "Energization: 1) Inform grid control room. 2) Close HV circuit breaker. "
     "3) Monitor inrush current — typically 8-12x rated for 0.5 seconds. "
     "4) Check secondary voltage — should be within ±5% of rated. "
     "5) Monitor oil and winding temperatures for 2 hours. "
     "6) Gradually apply load in 25% increments every 30 minutes."),

    ("SOP-EL-002", "DG Set Emergency Start Procedure", "diesel_generator",
     "Procedure for emergency manual start of diesel generator sets when AMF (Auto Mains Failure) panel fails. "
     "1) Verify mains supply is indeed lost — check incomer breaker status. "
     "2) At DG control panel, select 'LOCAL' mode. "
     "3) Check engine oil pressure gauge — should be zero (engine off). "
     "4) Check coolant level in expansion tank. "
     "5) Ensure fuel isolation valve is OPEN. "
     "6) Press START button — hold for maximum 10 seconds. "
     "7) If engine does not start, wait 30 seconds before retry. Maximum 3 attempts. "
     "8) Once running, check: RPM 1500 ±1%, voltage 415V ±5%, frequency 50 Hz ±0.5%. "
     "9) Close DG output breaker. 10) Transfer load using changeover switch. "
     "CAUTION: Never parallel DG with mains without synchronization equipment."),

    ("SOP-EL-003", "UPS Bypass Procedure", "ups",
     "Procedure for transferring UPS to maintenance bypass for servicing. "
     "1) Inform all affected departments of planned transfer. "
     "2) Verify static bypass is healthy — check bypass available LED. "
     "3) At UPS control panel, press 'Transfer to Bypass'. "
     "4) Verify output voltage remains stable — no break transfer. "
     "5) Confirm load is now on bypass path — inverter LED should be off. "
     "6) Open maintenance bypass switch (MBS). "
     "7) Confirm load transferred to MBS — now safe to isolate UPS. "
     "8) Switch off UPS inverter and rectifier. "
     "9) Open battery breaker for isolation. "
     "WARNING: Load is NOT protected during bypass. Minimize bypass duration."),

    ("SOP-HV-001", "Chiller Startup After Extended Shutdown", "chiller",
     "Procedure for starting chiller after extended shutdown (>72 hours). "
     "1) Verify chilled water system is filled and pressurized. "
     "2) Check condenser water system is operational. "
     "3) Energize chiller control panel — verify no pending alarms. "
     "4) Check oil sump heater has been ON for minimum 12 hours (prevents refrigerant migration). "
     "5) Verify compressor oil level in sight glass. "
     "6) Start condenser water pump, then chilled water pump. "
     "7) Verify water flow rates: CHW ≥ minimum flow, CDW ≥ minimum flow. "
     "8) Enable chiller auto-start from BMS. "
     "9) Monitor: suction pressure, discharge pressure, oil pressure differential. "
     "10) Allow 15-minute minimum run time before loading."),

    ("SOP-HV-002", "AHU Filter Replacement Procedure", "ahu",
     "Standard procedure for replacing AHU pre-filters and fine filters. "
     "1) Stop AHU from BMS or local panel. 2) Lock out electrical supply — LOTO. "
     "3) Open filter access panels. 4) Remove dirty filters — bag and dispose per waste protocol. "
     "5) Clean filter holding frames with damp cloth. "
     "6) Install new filters — check airflow direction arrows match. "
     "7) Pre-filters: MERV 8, replace every 3 months. Fine filters: MERV 13, replace every 6 months. "
     "8) Close access panels. 9) Remove LOTO. 10) Start AHU. "
     "11) Check filter DP after 1 hour — new filter: 50-80 Pa (pre), 100-150 Pa (fine). "
     "12) Record in filter replacement log."),

    ("SOP-ME-001", "Pump Alignment Procedure — Reverse Dial Indicator Method", "pump",
     "Procedure for aligning pump-motor coupling using reverse dial indicator method. "
     "Tools required: Dial indicator set, magnetic bases, feeler gauges, alignment shims. "
     "1) Disconnect coupling. Clean shaft surfaces. "
     "2) Mount dial indicators on pump shaft and motor shaft. "
     "3) Zero both indicators at 12 o'clock. "
     "4) Rotate shafts in 90° increments: 12, 3, 6, 9 o'clock. Record readings. "
     "5) Calculate angular and parallel misalignment: "
     "   Angular: (T-B)/2 and (L-R)/2, Parallel: (T+B)/2 and (L+R)/2 "
     "6) Adjust motor position using shims (vertical) and jackscrews (horizontal). "
     "7) Acceptable tolerance: Angular < 0.05mm, Parallel < 0.05mm. "
     "8) Re-connect coupling. Hand rotate to verify free movement. "
     "9) Run pump, measure vibration — should be < 2.5 mm/s."),

    ("SOP-ME-002", "Compressed Air System Leak Audit", "compressor",
     "Procedure for performing ultrasonic compressed air leak audit. "
     "Equipment: Ultrasonic leak detector (SDT 270 or equivalent), tagging materials. "
     "1) Ensure compressor is at normal operating pressure (7-8 bar). "
     "2) Systematically walk the compressed air distribution: main headers, branch lines, drops. "
     "3) Scan joints, couplings, valves, FRLs, cylinders, hoses using ultrasonic detector. "
     "4) When leak detected: tag location, estimate size (small/medium/large), log GPS/area. "
     "5) Estimate CFM loss per leak using detector dB reading and distance chart. "
     "6) Prioritize: Large leaks (>5 CFM) — fix within 24 hours. Medium (1-5 CFM) — fix within 1 week. "
     "7) Calculate total leak percentage: Total leak CFM / Total compressor CFM × 100. "
     "8) Target: <10% of total system capacity. Industry benchmark: 20-30% typical, <10% excellent."),

    ("SOP-EL-004", "Energy Meter Calibration Verification", "energy_meter",
     "Procedure for verifying energy meter calibration accuracy using reference standard. "
     "Equipment: Reference class 0.1 portable meter, CT test leads, current clamp. "
     "1) Connect reference meter in parallel with meter under test. "
     "2) Measure for minimum 15-minute interval at stable load. "
     "3) Compare kWh readings: Error % = (Test - Reference) / Reference × 100. "
     "4) Acceptable accuracy: Class 0.2S: ±0.2%, Class 0.5S: ±0.5%, Class 1.0: ±1.0%. "
     "5) If outside tolerance: check CT connections, PT ratio, meter programming. "
     "6) Re-program if necessary. Repeat verification. "
     "7) Record results in calibration certificate. Next verification: 12 months."),
]

INSPECTION_REPORT_TEMPLATES = [
    ("Thermographic Inspection Report — Electrical Systems",
     "Annual thermographic inspection of all HV/LV electrical installations. "
     "Equipment surveyed: Main transformers (5), PCC (1), MCC panels (3), "
     "distribution boards (34), VFD panels (11), UPS systems (8). "
     "Survey conditions: Ambient temperature {amb_temp}°C, load >60% on all circuits. "
     "Camera: FLIR T540, emissivity set to 0.95 for painted surfaces. "
     "\n\nFindings:\n"
     "CRITICAL (Immediate action): {critical_count} hot spots detected:\n{critical_findings}\n"
     "WARNING (Action within 1 week): {warning_count} elevated temperatures:\n{warning_findings}\n"
     "NORMAL: {normal_count} connections inspected — all within acceptable limits.\n"
     "\nRecommendations:\n"
     "1. Critical findings: Shut down affected equipment and re-torque connections immediately.\n"
     "2. Warning findings: Schedule re-torque during next planned maintenance window.\n"
     "3. Repeat survey after remediation to confirm corrections.\n"
     "4. Next annual thermographic survey due: {next_survey_date}"),

    ("Vibration Analysis Report — Rotating Equipment",
     "Quarterly vibration analysis of critical rotating equipment. "
     "Equipment surveyed: Pumps ({pump_count}), Motors ({motor_count}), "
     "Compressors ({comp_count}), AHU fans ({ahu_count}). "
     "Measurement: Tri-axial accelerometer, frequency range 10-1000 Hz. "
     "Standard: ISO 10816-3 for industrial machines. "
     "\n\nSummary:\n"
     "Zone A (Good): {zone_a} machines\n"
     "Zone B (Acceptable): {zone_b} machines\n"
     "Zone C (Alert): {zone_c} machines\n"
     "Zone D (Danger): {zone_d} machines\n"
     "\nZone C machines requiring attention:\n{zone_c_details}\n"
     "\nZone D machines requiring IMMEDIATE action:\n{zone_d_details}\n"
     "\nTrend Analysis: Compared with previous quarter readings, {trend_summary}"),

    ("Transformer Oil Analysis Report (DGA)",
     "Dissolved Gas Analysis (DGA) report for transformer {transformer_id}. "
     "Oil sample collected on {sample_date}. Tested at {lab_name} per IEC 60599. "
     "\nGas concentrations (ppm):\n"
     "Hydrogen (H2): {h2} ppm\n"
     "Methane (CH4): {ch4} ppm\n"
     "Ethane (C2H6): {c2h6} ppm\n"
     "Ethylene (C2H4): {c2h4} ppm\n"
     "Acetylene (C2H2): {c2h2} ppm\n"
     "Carbon Monoxide (CO): {co} ppm\n"
     "Carbon Dioxide (CO2): {co2} ppm\n"
     "TDCG (Total Dissolved Combustible Gas): {tdcg} ppm\n"
     "\nDuval Triangle Analysis: {duval_result}\n"
     "Rogers Ratio Analysis: {rogers_result}\n"
     "\nOil Quality:\n"
     "BDV (Breakdown Voltage): {bdv} kV (Min acceptable: 50 kV)\n"
     "Moisture Content: {moisture} ppm (Max acceptable: 30 ppm)\n"
     "Acidity: {acidity} mg KOH/g (Max acceptable: 0.5)\n"
     "\nConclusion: {conclusion}\n"
     "Recommendation: {recommendation}"),

    ("Energy Audit Report — Monthly",
     "Monthly energy consumption and efficiency report for {month} {year}. "
     "\nConsumption Summary:\n"
     "Total Grid Import: {total_kwh:,.0f} kWh\n"
     "Peak Demand: {peak_demand:,.0f} kW (Contract: {contract_demand:,.0f} kW)\n"
     "Average Power Factor: {avg_pf:.2f}\n"
     "DG Run Hours: {dg_hours:.0f} hours\n"
     "DG Fuel Consumed: {fuel_consumed:,.0f} liters\n"
     "DG Specific Fuel Consumption: {sfc:.2f} liters/kWh\n"
     "\nMajor Consumers:\n"
     "HVAC (Chillers + AHUs + Pumps): {hvac_pct:.0f}% ({hvac_kwh:,.0f} kWh)\n"
     "Production/Process: {prod_pct:.0f}% ({prod_kwh:,.0f} kWh)\n"
     "Lighting & Small Power: {light_pct:.0f}% ({light_kwh:,.0f} kWh)\n"
     "Compressed Air: {comp_pct:.0f}% ({comp_kwh:,.0f} kWh)\n"
     "UPS & IT Load: {ups_pct:.0f}% ({ups_kwh:,.0f} kWh)\n"
     "\nKPIs:\n"
     "Energy Performance Index (EPI): {epi:.1f} kWh/sq.ft/year\n"
     "Specific Energy Consumption: {sec:.2f} kWh/unit produced\n"
     "HVAC kW/TR: {kw_per_tr:.2f}\n"
     "\nEnergy Saving Opportunities Identified:\n{saving_opportunities}"),

    ("Shift Handover Log — Operations",
     "Shift: {shift_name}\nDate: {shift_date}\nSupervisor: {supervisor}\n"
     "\nPlant Status at Handover:\n"
     "Chillers Running: {chillers_running}/{chillers_total}\n"
     "DG Status: {dg_status}\n"
     "UPS Systems: {ups_status}\n"
     "Active Alerts: {active_alerts}\n"
     "\nEvents During Shift:\n{shift_events}\n"
     "\nPending Actions for Next Shift:\n{pending_actions}\n"
     "\nEquipment Status Changes:\n{status_changes}\n"
     "\nSafety Observations: {safety_obs}\n"
     "\nHandover accepted by: {next_supervisor}"),

    ("Commissioning Report — New Equipment",
     "Commissioning report for {equipment_name} ({equipment_id}). "
     "Date of commissioning: {commission_date}. "
     "Vendor engineer: {vendor_engineer}. Site engineer: {site_engineer}.\n"
     "\nPre-Commissioning Checks:\n{pre_checks}\n"
     "\nCommissioning Test Results:\n{test_results}\n"
     "\nPerformance Verification:\n{performance_data}\n"
     "\nPunch List Items:\n{punch_items}\n"
     "\nConclusion: Equipment commissioned successfully. "
     "Warranty period: {warranty_months} months from {commission_date}. "
     "First PM scheduled for: {first_pm_date}."),
]


# ============================================================
# Energy Time-Series Data Templates
# ============================================================

def generate_energy_readings(meter_id, meter_name, days=365):
    """Generate realistic energy time-series readings for a meter."""
    readings = []
    base_load = random.uniform(100, 2000)  # kW base
    now = timezone.now()

    for day_offset in range(days):
        dt = now - timedelta(days=day_offset)
        day_of_week = dt.weekday()
        month = dt.month

        # Seasonal factor (higher in summer for HVAC)
        seasonal = 1.0 + 0.15 * math.sin((month - 1) * math.pi / 6)

        # Weekday vs weekend
        day_factor = 1.0 if day_of_week < 5 else 0.6

        # Generate hourly readings for this day
        for hour in range(0, 24, 4):  # Every 4 hours
            # Time-of-day pattern
            if 8 <= hour <= 18:
                hour_factor = 1.0 + 0.2 * math.sin((hour - 8) * math.pi / 10)
            elif 6 <= hour < 8 or 18 < hour <= 22:
                hour_factor = 0.7
            else:
                hour_factor = 0.4

            load = base_load * seasonal * day_factor * hour_factor
            load *= random.uniform(0.9, 1.1)  # Random noise

            readings.append({
                "meter_id": meter_id,
                "meter_name": meter_name,
                "timestamp": (dt.replace(hour=hour, minute=0, second=0)).isoformat(),
                "power_kw": round(load, 1),
                "power_kva": round(load / random.uniform(0.85, 0.95), 1),
                "power_factor": round(random.uniform(0.85, 0.97), 3),
                "voltage_avg": round(random.uniform(225, 235), 1),
                "current_avg": round(load / 0.415 / random.uniform(0.85, 0.95), 1),
                "frequency": round(random.uniform(49.9, 50.1), 2),
                "energy_kwh_cumulative": round(base_load * 24 * (days - day_offset) + load * 4, 0),
            })

    return readings


# ============================================================
# Command
# ============================================================

class Command(BaseCommand):
    help = "Generate massive realistic industrial data for RAG pipeline"

    def add_arguments(self, parser):
        parser.add_argument("--clear", action="store_true", help="Clear existing alerts/maintenance before generating")
        parser.add_argument("--stats", action="store_true", help="Show current data statistics only")
        parser.add_argument("--skip-equipment", action="store_true", help="Skip equipment generation (use existing)")

    def handle(self, *args, **options):
        if options["stats"]:
            self._print_stats()
            return

        if options["clear"]:
            self.stdout.write("Clearing alerts, maintenance records...")
            Alert.objects.all().delete()
            MaintenanceRecord.objects.all().delete()
            # Also clear new tables if they exist
            self._clear_extra_tables()

        self.stdout.write(self.style.NOTICE("\n=== Generating Rich Industrial Data ===\n"))

        # Check if equipment exists
        eq_count = sum([
            Transformer.objects.count(), DieselGenerator.objects.count(),
            ElectricalPanel.objects.count(), UPS.objects.count(),
            Chiller.objects.count(), AHU.objects.count(),
            CoolingTower.objects.count(), Pump.objects.count(),
            Compressor.objects.count(), Motor.objects.count(),
            EnergyMeter.objects.count(),
        ])

        if eq_count == 0 and not options["skip_equipment"]:
            self.stdout.write(self.style.WARNING("No equipment found. Run 'python manage.py populate_industrial_db' first."))
            return

        self.stdout.write(f"Found {eq_count} equipment records.\n")

        # Gather all equipment for cross-referencing
        all_equipment = self._gather_all_equipment()

        # Generate data
        self._generate_alerts(all_equipment)
        self._generate_maintenance(all_equipment)
        self._generate_operational_documents(all_equipment)
        self._generate_energy_timeseries(all_equipment)
        self._generate_shift_logs(all_equipment)
        self._generate_work_orders(all_equipment)

        self.stdout.write(self.style.SUCCESS("\n=== Data Generation Complete ===\n"))
        self._print_stats()

    def _gather_all_equipment(self):
        """Collect all equipment into a unified list for cross-referencing."""
        equipment = []
        models_map = [
            (Transformer, "transformer"),
            (DieselGenerator, "diesel_generator"),
            (ElectricalPanel, "electrical_panel"),
            (UPS, "ups"),
            (Chiller, "chiller"),
            (AHU, "ahu"),
            (CoolingTower, "cooling_tower"),
            (Pump, "pump"),
            (Compressor, "compressor"),
            (Motor, "motor"),
            (EnergyMeter, "energy_meter"),
        ]

        for model, eq_type in models_map:
            for eq in model.objects.all():
                equipment.append({
                    "type": eq_type,
                    "id": eq.equipment_id,
                    "name": eq.name,
                    "status": eq.status,
                    "criticality": eq.criticality,
                    "building": eq.building,
                    "location": eq.location,
                    "health_score": eq.health_score,
                })

        return equipment

    def _generate_alerts(self, all_equipment):
        """Generate 500+ realistic alerts for ALL equipment types."""
        self.stdout.write("  Generating alerts...")
        alerts = []

        for eq in all_equipment:
            eq_type = eq["type"]
            templates = ALERT_TEMPLATES_BY_TYPE.get(eq_type, [])
            if not templates:
                continue

            # Number of alerts depends on health score (unhealthy = more alerts)
            health = eq["health_score"]
            if health < 50:
                n_alerts = random.randint(5, 12)
            elif health < 70:
                n_alerts = random.randint(2, 6)
            elif health < 85:
                n_alerts = random.randint(1, 3)
            else:
                n_alerts = random.randint(0, 2)

            for _ in range(n_alerts):
                tmpl = random.choice(templates)
                alert_type, severity, message, param, value_range, threshold, unit = tmpl

                value = None
                if value_range and isinstance(value_range, tuple):
                    value = _r(value_range[0], value_range[1])

                triggered = _ts(180)  # Last 6 months
                resolved = random.random() > 0.35  # 65% resolved
                ack = resolved or random.random() > 0.3

                alerts.append(Alert(
                    equipment_type=eq_type.replace("_", " ").title().replace(" ", ""),
                    equipment_id=eq["id"],
                    equipment_name=eq["name"],
                    severity=severity,
                    alert_type=alert_type,
                    message=message,
                    parameter=param or "",
                    value=value,
                    threshold=threshold,
                    unit=unit or "",
                    triggered_at=triggered,
                    acknowledged=ack,
                    acknowledged_by=random.choice(TECHNICIANS) if ack else "",
                    acknowledged_at=triggered + timedelta(minutes=random.randint(5, 120)) if ack else None,
                    resolved=resolved,
                    resolved_at=triggered + timedelta(hours=random.randint(1, 48)) if resolved else None,
                    notes=f"Investigated by {random.choice(TECHNICIANS)}. {'Root cause identified and corrected.' if resolved else 'Investigation in progress.'}" if random.random() > 0.4 else "",
                ))

        Alert.objects.bulk_create(alerts, batch_size=500)
        self.stdout.write(f"    Created {len(alerts)} alerts")

    def _generate_maintenance(self, all_equipment):
        """Generate 2000+ maintenance records for ALL equipment."""
        self.stdout.write("  Generating maintenance records...")
        records = []

        for eq in all_equipment:
            eq_type = eq["type"]
            work_templates = MAINT_WORK_BY_TYPE.get(eq_type, MAINT_WORK_GENERIC)

            # Number of maintenance records per equipment
            n_records = random.randint(3, 15)

            for _ in range(n_records):
                mtype = random.choice(["preventive", "corrective", "inspection", "breakdown"])
                work_list = work_templates.get(mtype, MAINT_WORK_GENERIC.get(mtype, MAINT_WORK_GENERIC["preventive"]))
                desc, work_done, parts = random.choice(work_list)

                scheduled = _ts(730)  # Last 2 years
                started = scheduled + timedelta(hours=random.randint(0, 48))
                completed = started + timedelta(hours=random.uniform(0.5, 72))
                downtime = (completed - started).total_seconds() / 3600 if mtype != "inspection" else 0

                records.append(MaintenanceRecord(
                    equipment_type=eq_type.replace("_", " ").title().replace(" ", ""),
                    equipment_id=eq["id"],
                    equipment_name=eq["name"],
                    maintenance_type=mtype if mtype != "breakdown" else "corrective",
                    description=desc,
                    work_done=work_done,
                    parts_replaced=parts,
                    cost=round(random.uniform(500, 150000), 2) if mtype != "inspection" else round(random.uniform(0, 5000), 2),
                    technician=random.choice(TECHNICIANS),
                    vendor=random.choice(VENDORS) if random.random() > 0.4 else "Internal Maintenance Team",
                    scheduled_date=scheduled,
                    started_at=started,
                    completed_at=completed,
                    downtime_hours=round(downtime, 1),
                ))

        MaintenanceRecord.objects.bulk_create(records, batch_size=500)
        self.stdout.write(f"    Created {len(records)} maintenance records")

    def _generate_operational_documents(self, all_equipment):
        """Generate operational documents: SOPs, inspection reports, shift logs."""
        self.stdout.write("  Generating operational documents...")

        # Create table if not exists
        self._ensure_operational_docs_table()

        docs = []

        # SOPs — one per template
        for sop_id, title, eq_type, content in SOP_TEMPLATES:
            docs.append({
                "doc_id": sop_id,
                "doc_type": "sop",
                "title": title,
                "equipment_type": eq_type,
                "content": content,
                "created_at": _ts(365).isoformat(),
                "version": f"Rev {random.randint(1, 5)}",
                "author": random.choice(TECHNICIANS),
                "department": "Operations & Maintenance",
            })

        # Generate many inspection reports
        for i in range(200):
            eq = random.choice(all_equipment)
            doc_type = random.choice(["inspection_report", "incident_report", "commissioning_report", "calibration_certificate"])

            if doc_type == "inspection_report":
                title = f"Inspection Report — {eq['name']} ({eq['id']})"
                content = self._generate_inspection_content(eq)
            elif doc_type == "incident_report":
                title = f"Incident Report — {eq['name']}"
                content = self._generate_incident_content(eq)
            elif doc_type == "commissioning_report":
                title = f"Commissioning Report — {eq['name']}"
                content = self._generate_commissioning_content(eq)
            else:
                title = f"Calibration Certificate — {eq['name']} ({eq['id']})"
                content = self._generate_calibration_content(eq)

            docs.append({
                "doc_id": f"DOC-{doc_type[:3].upper()}-{i+1:04d}",
                "doc_type": doc_type,
                "title": title,
                "equipment_type": eq["type"],
                "equipment_id": eq["id"],
                "content": content,
                "created_at": _ts(730).isoformat(),
                "version": "Rev 1",
                "author": random.choice(TECHNICIANS),
                "department": random.choice(["Operations", "Maintenance", "Safety", "Quality"]),
            })

        # Generate energy audit documents
        for month_offset in range(24):  # 2 years of monthly reports
            dt = timezone.now() - timedelta(days=month_offset * 30)
            total_kwh = random.uniform(400000, 800000)
            hvac_pct = random.uniform(35, 50)
            prod_pct = random.uniform(25, 35)
            light_pct = random.uniform(8, 15)
            comp_pct = random.uniform(5, 10)
            ups_pct = 100 - hvac_pct - prod_pct - light_pct - comp_pct

            docs.append({
                "doc_id": f"DOC-EAR-{dt.strftime('%Y%m')}",
                "doc_type": "energy_audit",
                "title": f"Energy Audit Report — {dt.strftime('%B %Y')}",
                "equipment_type": "energy_meter",
                "content": (
                    f"Monthly energy consumption report for {dt.strftime('%B %Y')}. "
                    f"Total Grid Import: {total_kwh:,.0f} kWh. "
                    f"Peak Demand: {random.uniform(1800, 2800):,.0f} kW (Contract: 3000 kW). "
                    f"Average Power Factor: {random.uniform(0.88, 0.96):.2f}. "
                    f"DG Run Hours: {random.uniform(10, 120):.0f} hours. "
                    f"DG Fuel Consumed: {random.uniform(500, 5000):,.0f} liters. "
                    f"\nBreakdown: HVAC {hvac_pct:.0f}% ({total_kwh*hvac_pct/100:,.0f} kWh), "
                    f"Production {prod_pct:.0f}% ({total_kwh*prod_pct/100:,.0f} kWh), "
                    f"Lighting {light_pct:.0f}% ({total_kwh*light_pct/100:,.0f} kWh), "
                    f"Compressed Air {comp_pct:.0f}% ({total_kwh*comp_pct/100:,.0f} kWh), "
                    f"UPS/IT {ups_pct:.0f}% ({total_kwh*ups_pct/100:,.0f} kWh). "
                    f"EPI: {random.uniform(80, 150):.1f} kWh/sq.ft/year. "
                    f"HVAC kW/TR: {random.uniform(0.8, 1.3):.2f}. "
                    f"Savings identified: VFD on AHU fans ({random.uniform(5000, 20000):,.0f} kWh/month), "
                    f"Chiller sequencing optimization ({random.uniform(3000, 15000):,.0f} kWh/month), "
                    f"Compressed air leak repair ({random.uniform(2000, 8000):,.0f} kWh/month)."
                ),
                "created_at": dt.isoformat(),
                "version": "Rev 1",
                "author": "Energy Manager",
                "department": "Energy & Sustainability",
            })

        # Bulk insert into operational_documents table
        self._insert_operational_docs(docs)
        self.stdout.write(f"    Created {len(docs)} operational documents")

    def _generate_inspection_content(self, eq):
        """Generate realistic inspection report content."""
        findings = random.choice([
            "All parameters within acceptable limits. No action required.",
            f"Minor corrosion found on {eq['name']} mounting bolts. Recommend anti-corrosion treatment during next PM.",
            f"Slight vibration increase detected on {eq['name']}. Current: {_r(2.5, 4.5)} mm/s (limit: 4.5 mm/s). Monitor closely.",
            f"Thermal imaging shows connection temperature of {_r(55, 75)}°C at terminal block. Re-torque recommended.",
            f"Oil analysis for {eq['name']} shows elevated iron content ({_r(20, 50)} ppm). Schedule wear analysis.",
            f"Bearing noise detected during inspection. Recommend replacement at next scheduled maintenance.",
            f"Insulation resistance measured at {_r(100, 500)} MΩ — acceptable but trending downward. Monitor quarterly.",
        ])

        return (
            f"Equipment Inspection Report\n"
            f"Equipment: {eq['name']} ({eq['id']})\n"
            f"Location: {eq['location']}, {eq['building']}\n"
            f"Date: {_ts(90).strftime('%Y-%m-%d')}\n"
            f"Inspector: {random.choice(TECHNICIANS)}\n"
            f"Type: {'Routine quarterly' if random.random() > 0.3 else 'Special'} inspection\n"
            f"\nEquipment Status: {eq['status'].title()}\n"
            f"Health Score: {eq['health_score']}%\n"
            f"\nFindings:\n{findings}\n"
            f"\nRecommendation: {'No immediate action' if eq['health_score'] > 80 else 'Schedule corrective maintenance'}"
        )

    def _generate_incident_content(self, eq):
        """Generate incident report content."""
        incidents = [
            (f"{eq['name']} tripped unexpectedly during normal operation",
             "Overcurrent protection operated",
             "Found loose connection at terminal block causing arcing. Re-terminated and torqued.",
             "Re-inspect all similar equipment connections within 1 week."),
            (f"{eq['name']} performance degradation noticed during routine check",
             "Output parameters below design values",
             "Fouling/blockage in system components. Cleaned and restored to normal operation.",
             "Increase cleaning frequency from quarterly to monthly."),
            (f"Unusual noise reported from {eq['name']} by operator",
             "Abnormal bearing noise detected at DE end",
             "Bearing showing early signs of outer race defect. Replaced bearing.",
             "Add this equipment to vibration monitoring program."),
            (f"Oil leak detected at base of {eq['name']}",
             "Seal deterioration due to age and thermal cycling",
             "Replaced seal and cleaned spill area. Environmental team notified per protocol.",
             "Schedule seal inspection for all similar-age equipment."),
        ]
        incident = random.choice(incidents)
        return (
            f"Incident Report\n"
            f"Equipment: {eq['name']} ({eq['id']})\n"
            f"Location: {eq['location']}\n"
            f"Date/Time: {_ts(180).strftime('%Y-%m-%d %H:%M')}\n"
            f"Reported by: {random.choice(TECHNICIANS)}\n"
            f"Severity: {'Critical' if eq['criticality'] == 'critical' else 'Major' if eq['criticality'] == 'high' else 'Minor'}\n"
            f"\nDescription: {incident[0]}\n"
            f"Root Cause: {incident[1]}\n"
            f"Action Taken: {incident[2]}\n"
            f"Corrective Action: {incident[3]}\n"
            f"Downtime: {_r(0.5, 24)} hours\n"
            f"Cost Impact: ₹{random.randint(5000, 200000):,}"
        )

    def _generate_commissioning_content(self, eq):
        """Generate commissioning report content."""
        return (
            f"Equipment Commissioning Report\n"
            f"Equipment: {eq['name']} ({eq['id']})\n"
            f"Location: {eq['location']}, {eq['building']}\n"
            f"Commissioning Date: {_ts(365).strftime('%Y-%m-%d')}\n"
            f"Vendor Engineer: {random.choice(VENDORS)}\n"
            f"Site Engineer: {random.choice(TECHNICIANS)}\n"
            f"\nPre-Commissioning Checks: All mechanical and electrical installations verified per drawings. "
            f"Foundation bolts torqued. Alignment within tolerance. Wiring checked against schematics.\n"
            f"\nTest Results:\n"
            f"- No-load test: PASSED\n"
            f"- Full-load test: PASSED (operated at {_r(95, 105)}% of rated capacity)\n"
            f"- Safety interlocks: All {random.randint(3, 8)} interlocks tested and functional\n"
            f"- Control system: All {random.randint(5, 20)} I/O points verified\n"
            f"- Performance: Meeting design specifications\n"
            f"\nWarranty: {random.choice([12, 18, 24])} months from commissioning date\n"
            f"First PM scheduled: {(_ts(0) + timedelta(days=random.randint(60, 180))).strftime('%Y-%m-%d')}"
        )

    def _generate_calibration_content(self, eq):
        """Generate calibration certificate content."""
        return (
            f"Calibration Certificate\n"
            f"Equipment: {eq['name']} ({eq['id']})\n"
            f"Location: {eq['location']}\n"
            f"Calibration Date: {_ts(180).strftime('%Y-%m-%d')}\n"
            f"Calibrated by: {random.choice(VENDORS)}\n"
            f"Reference Standard: NABL accredited reference instrument\n"
            f"Traceability: NPL India\n"
            f"\nResults:\n"
            f"- Zero check: PASSED (deviation < 0.1%)\n"
            f"- Span check at 25%: Reading {_r(24.8, 25.2)}%, Error: {_r(-0.2, 0.2)}%\n"
            f"- Span check at 50%: Reading {_r(49.8, 50.2)}%, Error: {_r(-0.2, 0.2)}%\n"
            f"- Span check at 75%: Reading {_r(74.8, 75.2)}%, Error: {_r(-0.3, 0.3)}%\n"
            f"- Span check at 100%: Reading {_r(99.7, 100.3)}%, Error: {_r(-0.3, 0.3)}%\n"
            f"\nAccuracy: Within ±{random.choice(['0.2', '0.5', '1.0'])}% of full scale\n"
            f"Status: {'PASS — Within tolerance' if random.random() > 0.1 else 'MARGINAL — Recalibrate in 6 months'}\n"
            f"Next calibration due: {(_ts(0) + timedelta(days=365)).strftime('%Y-%m-%d')}"
        )

    def _generate_energy_timeseries(self, all_equipment):
        """Generate energy time-series data."""
        self.stdout.write("  Generating energy time-series data...")

        self._ensure_energy_readings_table()

        meters = [eq for eq in all_equipment if eq["type"] == "energy_meter"]
        # Pick top 10 meters for time-series (main incomers + major feeders)
        selected = meters[:10] if len(meters) >= 10 else meters

        total_readings = 0
        for meter in selected:
            readings = generate_energy_readings(meter["id"], meter["name"], days=180)
            self._insert_energy_readings(readings)
            total_readings += len(readings)

        self.stdout.write(f"    Created {total_readings} energy readings across {len(selected)} meters")

    def _generate_shift_logs(self, all_equipment):
        """Generate shift handover logs."""
        self.stdout.write("  Generating shift logs...")

        self._ensure_shift_logs_table()

        logs = []
        now = timezone.now()

        for day_offset in range(180):  # 6 months
            dt = now - timedelta(days=day_offset)

            for shift_name in SHIFTS:
                supervisor = random.choice(SHIFT_SUPERVISORS)
                next_supervisor = random.choice([s for s in SHIFT_SUPERVISORS if s != supervisor])

                # Generate random events
                n_events = random.randint(1, 6)
                events = []
                for _ in range(n_events):
                    eq = random.choice(all_equipment)
                    event_types = [
                        f"{eq['name']} started at {random.randint(0,23):02d}:{random.randint(0,59):02d}",
                        f"{eq['name']} stopped for maintenance",
                        f"Alert acknowledged on {eq['name']} — {random.choice(['temperature high', 'vibration elevated', 'pressure low'])}",
                        f"{eq['name']} tripped — reset and restarted after investigation",
                        f"Routine rounds completed — all parameters normal in {random.choice(BUILDINGS)}",
                        f"Vendor visit for {eq['name']} — {random.choice(['annual service', 'warranty repair', 'inspection'])}",
                        f"Power fluctuation observed at {random.randint(0,23):02d}:{random.randint(0,59):02d} — lasted {random.randint(1,30)} seconds",
                    ]
                    events.append(random.choice(event_types))

                pending_actions = random.choice([
                    "Monitor transformer TR-MAIN-01 oil temperature — elevated since morning.",
                    "Chiller CH-03 condenser cleaning scheduled for tomorrow.",
                    "Pump PMP-CHW-02 bearing replacement parts expected tomorrow.",
                    "Compressor COMP-05 VFD showing intermittent fault — vendor notified.",
                    "No pending actions.",
                    "DG-02 fuel top-up needed before next test.",
                    "AHU filter replacement team arriving at 09:00 tomorrow.",
                ])

                active_alerts_count = random.randint(0, 8)

                logs.append({
                    "log_id": f"SHIFT-{dt.strftime('%Y%m%d')}-{shift_name[:3].upper()}",
                    "shift_date": dt.strftime("%Y-%m-%d"),
                    "shift_name": shift_name,
                    "supervisor": supervisor,
                    "next_supervisor": next_supervisor,
                    "active_alerts": active_alerts_count,
                    "events": "\n".join(f"- {e}" for e in events),
                    "pending_actions": pending_actions,
                    "notes": f"Overall plant status: {'Normal' if active_alerts_count < 3 else 'Elevated alert state'}. "
                             f"{'All systems running smoothly.' if random.random() > 0.3 else 'Monitoring ongoing issues.'}",
                    "created_at": dt.isoformat(),
                })

        self._insert_shift_logs(logs)
        self.stdout.write(f"    Created {len(logs)} shift log entries")

    def _generate_work_orders(self, all_equipment):
        """Generate work orders."""
        self.stdout.write("  Generating work orders...")

        self._ensure_work_orders_table()

        work_orders = []
        priorities = ["emergency", "urgent", "high", "normal", "low"]
        statuses = ["open", "in_progress", "on_hold", "completed", "cancelled"]

        for i in range(800):
            eq = random.choice(all_equipment)
            priority = random.choices(priorities, weights=[5, 10, 20, 45, 20])[0]
            status = random.choices(statuses, weights=[15, 20, 5, 55, 5])[0]

            work_type = random.choice(["preventive", "corrective", "predictive", "emergency"])
            descriptions = {
                "preventive": f"Scheduled PM for {eq['name']} — quarterly service as per maintenance calendar.",
                "corrective": f"Corrective maintenance for {eq['name']} — {random.choice(['bearing replacement', 'seal repair', 'filter change', 'sensor calibration', 'connection tightening'])}.",
                "predictive": f"Predictive maintenance for {eq['name']} based on {random.choice(['vibration trend', 'thermal imaging', 'oil analysis', 'current signature analysis'])} results.",
                "emergency": f"Emergency repair for {eq['name']} — {random.choice(['tripped on fault', 'leak detected', 'unusual noise', 'performance degradation'])}.",
            }

            created = _ts(365)
            due = created + timedelta(days=random.randint(1, 30))
            completed_at = None
            if status == "completed":
                completed_at = (created + timedelta(days=random.randint(1, 14))).isoformat()

            work_orders.append({
                "wo_id": f"WO-{i+1:05d}",
                "equipment_id": eq["id"],
                "equipment_name": eq["name"],
                "equipment_type": eq["type"],
                "work_type": work_type,
                "priority": priority,
                "status": status,
                "description": descriptions[work_type],
                "assigned_to": random.choice(TECHNICIANS),
                "vendor": random.choice(VENDORS) if random.random() > 0.6 else "",
                "estimated_hours": round(random.uniform(0.5, 48), 1),
                "actual_hours": round(random.uniform(0.5, 72), 1) if status == "completed" else None,
                "estimated_cost": round(random.uniform(1000, 200000), 2),
                "actual_cost": round(random.uniform(500, 250000), 2) if status == "completed" else None,
                "created_at": created.isoformat(),
                "due_date": due.strftime("%Y-%m-%d"),
                "completed_at": completed_at,
                "notes": random.choice([
                    "", "Spare parts ordered.", "Waiting for vendor availability.",
                    "Completed ahead of schedule.", "Extended due to additional findings.",
                    "Parts in stock. Ready to execute.", "Requires plant shutdown for access.",
                ]),
            })

        self._insert_work_orders(work_orders)
        self.stdout.write(f"    Created {len(work_orders)} work orders")

    # ============================================================
    # Database helpers — raw SQL for new tables
    # ============================================================

    def _ensure_operational_docs_table(self):
        with connection.cursor() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS operational_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT UNIQUE NOT NULL,
                    doc_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    equipment_type TEXT,
                    equipment_id TEXT,
                    content TEXT NOT NULL,
                    created_at TEXT,
                    version TEXT,
                    author TEXT,
                    department TEXT
                )
            """)

    def _insert_operational_docs(self, docs):
        with connection.cursor() as c:
            for doc in docs:
                c.execute("""
                    INSERT OR REPLACE INTO operational_documents
                    (doc_id, doc_type, title, equipment_type, equipment_id, content, created_at, version, author, department)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, [
                    doc["doc_id"], doc["doc_type"], doc["title"],
                    doc.get("equipment_type", ""), doc.get("equipment_id", ""),
                    doc["content"], doc["created_at"], doc["version"],
                    doc["author"], doc["department"],
                ])

    def _ensure_energy_readings_table(self):
        with connection.cursor() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS energy_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meter_id TEXT NOT NULL,
                    meter_name TEXT,
                    timestamp TEXT NOT NULL,
                    power_kw REAL,
                    power_kva REAL,
                    power_factor REAL,
                    voltage_avg REAL,
                    current_avg REAL,
                    frequency REAL,
                    energy_kwh_cumulative REAL
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_energy_meter ON energy_readings(meter_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_energy_ts ON energy_readings(timestamp)")

    def _insert_energy_readings(self, readings):
        with connection.cursor() as c:
            for r in readings:
                c.execute("""
                    INSERT INTO energy_readings
                    (meter_id, meter_name, timestamp, power_kw, power_kva, power_factor, voltage_avg, current_avg, frequency, energy_kwh_cumulative)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, [
                    r["meter_id"], r["meter_name"], r["timestamp"],
                    r["power_kw"], r["power_kva"], r["power_factor"],
                    r["voltage_avg"], r["current_avg"], r["frequency"],
                    r["energy_kwh_cumulative"],
                ])

    def _ensure_shift_logs_table(self):
        with connection.cursor() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS shift_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    log_id TEXT UNIQUE NOT NULL,
                    shift_date TEXT NOT NULL,
                    shift_name TEXT NOT NULL,
                    supervisor TEXT,
                    next_supervisor TEXT,
                    active_alerts INTEGER,
                    events TEXT,
                    pending_actions TEXT,
                    notes TEXT,
                    created_at TEXT
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_shift_date ON shift_logs(shift_date)")

    def _insert_shift_logs(self, logs):
        with connection.cursor() as c:
            for log in logs:
                c.execute("""
                    INSERT OR REPLACE INTO shift_logs
                    (log_id, shift_date, shift_name, supervisor, next_supervisor, active_alerts, events, pending_actions, notes, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, [
                    log["log_id"], log["shift_date"], log["shift_name"],
                    log["supervisor"], log["next_supervisor"], log["active_alerts"],
                    log["events"], log["pending_actions"], log["notes"], log["created_at"],
                ])

    def _ensure_work_orders_table(self):
        with connection.cursor() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS work_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    wo_id TEXT UNIQUE NOT NULL,
                    equipment_id TEXT NOT NULL,
                    equipment_name TEXT,
                    equipment_type TEXT,
                    work_type TEXT,
                    priority TEXT,
                    status TEXT,
                    description TEXT,
                    assigned_to TEXT,
                    vendor TEXT,
                    estimated_hours REAL,
                    actual_hours REAL,
                    estimated_cost REAL,
                    actual_cost REAL,
                    created_at TEXT,
                    due_date TEXT,
                    completed_at TEXT,
                    notes TEXT
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_wo_equip ON work_orders(equipment_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_wo_status ON work_orders(status)")

    def _insert_work_orders(self, work_orders):
        with connection.cursor() as c:
            for wo in work_orders:
                c.execute("""
                    INSERT OR REPLACE INTO work_orders
                    (wo_id, equipment_id, equipment_name, equipment_type, work_type, priority, status,
                     description, assigned_to, vendor, estimated_hours, actual_hours,
                     estimated_cost, actual_cost, created_at, due_date, completed_at, notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, [
                    wo["wo_id"], wo["equipment_id"], wo["equipment_name"], wo["equipment_type"],
                    wo["work_type"], wo["priority"], wo["status"], wo["description"],
                    wo["assigned_to"], wo["vendor"], wo["estimated_hours"], wo["actual_hours"],
                    wo["estimated_cost"], wo["actual_cost"], wo["created_at"],
                    wo["due_date"], wo["completed_at"], wo["notes"],
                ])

    def _clear_extra_tables(self):
        """Clear the extra tables if they exist."""
        for table in ["operational_documents", "energy_readings", "shift_logs", "work_orders"]:
            try:
                with connection.cursor() as c:
                    c.execute(f"DELETE FROM {table}")
            except Exception:
                pass

    def _print_stats(self):
        """Print comprehensive data statistics."""
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write("  DATA STATISTICS")
        self.stdout.write("=" * 60)

        models = [
            ("Transformers", Transformer),
            ("Diesel Generators", DieselGenerator),
            ("Electrical Panels", ElectricalPanel),
            ("UPS Systems", UPS),
            ("Chillers", Chiller),
            ("AHUs", AHU),
            ("Cooling Towers", CoolingTower),
            ("Pumps", Pump),
            ("Compressors", Compressor),
            ("Motors", Motor),
            ("Energy Meters", EnergyMeter),
        ]

        self.stdout.write("\n  Equipment:")
        eq_total = 0
        for name, model in models:
            count = model.objects.count()
            eq_total += count
            self.stdout.write(f"    {name}: {count}")
        self.stdout.write(f"    TOTAL Equipment: {eq_total}")

        self.stdout.write(f"\n  Alerts: {Alert.objects.count()}")
        self.stdout.write(f"    Unresolved: {Alert.objects.filter(resolved=False).count()}")
        self.stdout.write(f"    Critical: {Alert.objects.filter(severity='critical').count()}")

        self.stdout.write(f"\n  Maintenance Records: {MaintenanceRecord.objects.count()}")

        # Extra tables
        for table, label in [
            ("operational_documents", "Operational Documents"),
            ("energy_readings", "Energy Readings"),
            ("shift_logs", "Shift Logs"),
            ("work_orders", "Work Orders"),
        ]:
            try:
                with connection.cursor() as c:
                    c.execute(f"SELECT COUNT(*) FROM {table}")
                    count = c.fetchone()[0]
                    self.stdout.write(f"  {label}: {count}")
            except Exception:
                self.stdout.write(f"  {label}: (table not created yet)")

        self.stdout.write("\n" + "=" * 60)
