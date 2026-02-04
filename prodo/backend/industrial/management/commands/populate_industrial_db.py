"""
Management command to populate the industrial energy database with dummy data.

Creates hundreds of equipment records across all categories:
- Transformers (20+)
- Diesel Generators (15+)
- Electrical Panels (50+)
- UPS Systems (25+)
- Chillers (10+)
- AHUs (40+)
- Cooling Towers (8+)
- Pumps (60+)
- Compressors (15+)
- Motors (80+)
- Energy Meters (100+)
- Alerts (50+)
- Maintenance Records (200+)

Usage:
    python manage.py populate_industrial_db
    python manage.py populate_industrial_db --clear  # Clear existing data first
"""

import random
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils import timezone
from industrial.models import (
    Transformer,
    DieselGenerator,
    ElectricalPanel,
    UPS,
    Chiller,
    AHU,
    CoolingTower,
    Pump,
    Compressor,
    Motor,
    EnergyMeter,
    Alert,
    MaintenanceRecord,
)


# ============================================================
# Data Generation Helpers
# ============================================================

BUILDINGS = ["Main Plant", "Block A", "Block B", "Utility Building", "Admin Building", "Warehouse"]
FLOORS = ["Ground Floor", "First Floor", "Second Floor", "Basement", "Terrace"]
ZONES = ["Zone 1", "Zone 2", "Zone 3", "Production Area", "Utility Area", "HVAC Plant Room"]

MANUFACTURERS = {
    "transformer": ["ABB", "Siemens", "Schneider Electric", "Crompton Greaves", "BHEL", "Kirloskar"],
    "dg": ["Cummins", "Caterpillar", "Kirloskar", "Ashok Leyland", "Mahindra Powerol", "Greaves"],
    "panel": ["ABB", "Siemens", "Schneider Electric", "L&T", "Havells", "Legrand"],
    "ups": ["APC", "Emerson", "Eaton", "Delta", "Luminous", "Microtek"],
    "chiller": ["Carrier", "Trane", "York", "Daikin", "Blue Star", "Voltas"],
    "ahu": ["Carrier", "Trane", "Blue Star", "Voltas", "Daikin", "Johnson Controls"],
    "pump": ["Grundfos", "KSB", "Kirloskar", "CRI", "Crompton", "Wilo"],
    "compressor": ["Atlas Copco", "Ingersoll Rand", "Kaeser", "Elgi", "Chicago Pneumatic"],
    "motor": ["ABB", "Siemens", "Crompton", "Kirloskar", "Havells", "Bharat Bijlee"],
    "meter": ["Schneider Electric", "ABB", "L&T", "Secure Meters", "HPL", "Elmeasure"],
}


def random_date(start_year=2018, end_year=2024):
    """Generate a random date between start and end year."""
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)


def random_status():
    """Generate weighted random status."""
    choices = [
        ("running", 70),
        ("stopped", 10),
        ("standby", 10),
        ("maintenance", 5),
        ("fault", 3),
        ("offline", 2),
    ]
    total = sum(w for _, w in choices)
    r = random.randint(1, total)
    cumulative = 0
    for status, weight in choices:
        cumulative += weight
        if r <= cumulative:
            return status
    return "running"


def random_health():
    """Generate weighted random health score."""
    # Most equipment healthy, some degraded, few critical
    if random.random() < 0.7:
        return random.randint(85, 100)
    elif random.random() < 0.9:
        return random.randint(60, 84)
    else:
        return random.randint(30, 59)


class Command(BaseCommand):
    help = "Populate the industrial energy database with dummy data"

    def add_arguments(self, parser):
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Clear existing data before populating",
        )

    def handle(self, *args, **options):
        if options["clear"]:
            self.stdout.write("Clearing existing data...")
            self.clear_data()

        self.stdout.write("Populating industrial energy database...")

        self.create_transformers()
        self.create_diesel_generators()
        self.create_electrical_panels()
        self.create_ups_systems()
        self.create_chillers()
        self.create_ahus()
        self.create_cooling_towers()
        self.create_pumps()
        self.create_compressors()
        self.create_motors()
        self.create_energy_meters()
        self.create_alerts()
        self.create_maintenance_records()

        self.stdout.write(self.style.SUCCESS("Database populated successfully!"))
        self.print_summary()

    def clear_data(self):
        """Clear all existing data."""
        models = [
            Transformer, DieselGenerator, ElectricalPanel, UPS,
            Chiller, AHU, CoolingTower, Pump, Compressor, Motor,
            EnergyMeter, Alert, MaintenanceRecord,
        ]
        for model in models:
            count = model.objects.count()
            model.objects.all().delete()
            self.stdout.write(f"  Deleted {count} {model.__name__} records")

    def create_transformers(self):
        """Create transformer records."""
        self.stdout.write("  Creating transformers...")
        transformers = []

        # Main power transformers
        capacities = [1000, 1500, 2000, 2500, 3000]
        for i, capacity in enumerate(capacities, 1):
            transformers.append(Transformer(
                equipment_id=f"TR-MAIN-{i:02d}",
                name=f"Main Transformer {i}",
                description=f"{capacity} kVA Main Distribution Transformer",
                location=f"{random.choice(BUILDINGS)} - Substation",
                building=random.choice(BUILDINGS),
                floor="Ground Floor",
                zone="Substation",
                manufacturer=random.choice(MANUFACTURERS["transformer"]),
                model_number=f"DT-{capacity}",
                installation_date=random_date(2018, 2022),
                status=random_status(),
                criticality="critical",
                health_score=random_health(),
                transformer_type="distribution",
                capacity_kva=capacity,
                primary_voltage=11000,
                secondary_voltage=433,
                vector_group="Dyn11",
                impedance_percent=random.uniform(4.5, 6.5),
                cooling_type="ONAN",
                load_percent=random.uniform(40, 85),
                oil_temperature=random.uniform(45, 75),
                winding_temperature=random.uniform(50, 85),
                input_voltage=random.uniform(10800, 11200),
                output_voltage=random.uniform(415, 440),
                current_load_kw=capacity * 0.8 * random.uniform(0.4, 0.85),
                running_hours=random.uniform(10000, 50000),
            ))

        # Dry type transformers
        for i in range(1, 16):
            capacity = random.choice([100, 150, 200, 250, 315, 400, 500])
            transformers.append(Transformer(
                equipment_id=f"TR-DRY-{i:02d}",
                name=f"Dry Type Transformer {i}",
                description=f"{capacity} kVA Dry Type Transformer for {random.choice(['HVAC', 'Lighting', 'Production', 'UPS'])}",
                location=random.choice(BUILDINGS),
                building=random.choice(BUILDINGS),
                floor=random.choice(FLOORS),
                zone=random.choice(ZONES),
                manufacturer=random.choice(MANUFACTURERS["transformer"]),
                model_number=f"DT-DRY-{capacity}",
                installation_date=random_date(2019, 2023),
                status=random_status(),
                criticality=random.choice(["high", "medium"]),
                health_score=random_health(),
                transformer_type="dry_type",
                capacity_kva=capacity,
                primary_voltage=433,
                secondary_voltage=230 if capacity < 200 else 433,
                impedance_percent=random.uniform(3.5, 5.5),
                cooling_type="AN",
                load_percent=random.uniform(30, 80),
                winding_temperature=random.uniform(40, 70),
                current_load_kw=capacity * 0.8 * random.uniform(0.3, 0.8),
                running_hours=random.uniform(5000, 30000),
            ))

        Transformer.objects.bulk_create(transformers)
        self.stdout.write(f"    Created {len(transformers)} transformers")

    def create_diesel_generators(self):
        """Create diesel generator records."""
        self.stdout.write("  Creating diesel generators...")
        dgs = []

        capacities = [500, 750, 1000, 1250, 1500, 2000]
        for i, capacity in enumerate(capacities, 1):
            dgs.append(DieselGenerator(
                equipment_id=f"DG-{i:02d}",
                name=f"DG Set {i}",
                description=f"{capacity} kVA Diesel Generator for Emergency Power",
                location="DG Yard",
                building="Utility Building",
                floor="Ground Floor",
                zone="DG Room",
                manufacturer=random.choice(MANUFACTURERS["dg"]),
                model_number=f"DG-{capacity}KVA",
                installation_date=random_date(2018, 2022),
                status=random.choice(["standby", "standby", "standby", "running", "maintenance"]),
                criticality="critical",
                health_score=random_health(),
                capacity_kva=capacity,
                capacity_kw=capacity * 0.8,
                voltage=433,
                frequency=50,
                power_factor=0.8,
                phases=3,
                engine_make=random.choice(["Cummins", "Kirloskar", "Caterpillar"]),
                fuel_tank_capacity=random.choice([1000, 2000, 3000, 5000]),
                current_load_kw=0 if random.random() > 0.2 else capacity * 0.8 * random.uniform(0.3, 0.7),
                load_percent=0 if random.random() > 0.2 else random.uniform(30, 70),
                fuel_level_percent=random.uniform(50, 100),
                coolant_temperature=random.uniform(25, 45) if random.random() > 0.2 else random.uniform(70, 90),
                oil_pressure=random.uniform(3.5, 5.0),
                battery_voltage=random.uniform(24, 28),
                total_run_hours=random.uniform(500, 5000),
                total_energy_kwh=random.uniform(10000, 100000),
                running_hours=random.uniform(500, 5000),
            ))

        # Smaller DGs for specific areas
        for i in range(7, 16):
            capacity = random.choice([125, 180, 250, 320, 380])
            dgs.append(DieselGenerator(
                equipment_id=f"DG-{i:02d}",
                name=f"DG Set {i} - {random.choice(['IT Block', 'Data Center', 'Critical Load'])}",
                description=f"{capacity} kVA Diesel Generator",
                location=random.choice(BUILDINGS),
                building=random.choice(BUILDINGS),
                floor="Ground Floor",
                zone="DG Room",
                manufacturer=random.choice(MANUFACTURERS["dg"]),
                model_number=f"DG-{capacity}KVA",
                installation_date=random_date(2020, 2024),
                status=random.choice(["standby", "standby", "maintenance"]),
                criticality="high",
                health_score=random_health(),
                capacity_kva=capacity,
                capacity_kw=capacity * 0.8,
                voltage=433,
                frequency=50,
                power_factor=0.8,
                phases=3,
                fuel_tank_capacity=random.choice([500, 1000]),
                fuel_level_percent=random.uniform(60, 100),
                total_run_hours=random.uniform(100, 2000),
                running_hours=random.uniform(100, 2000),
            ))

        DieselGenerator.objects.bulk_create(dgs)
        self.stdout.write(f"    Created {len(dgs)} diesel generators")

    def create_electrical_panels(self):
        """Create electrical panel records."""
        self.stdout.write("  Creating electrical panels...")
        panels = []

        # Main panels (PCC, MCC)
        panel_configs = [
            ("PCC", "Power Control Center", 3200, "pcc"),
            ("MCC-1", "Motor Control Center 1 - HVAC", 2500, "mcc"),
            ("MCC-2", "Motor Control Center 2 - Production", 2000, "mcc"),
            ("MCC-3", "Motor Control Center 3 - Pumps", 1600, "mcc"),
            ("PMCC", "Power & Motor Control Center", 2500, "pmcc"),
            ("APFC-1", "Auto Power Factor Panel 1", 1000, "apfc"),
            ("APFC-2", "Auto Power Factor Panel 2", 800, "apfc"),
        ]

        for eq_id, name, rating, ptype in panel_configs:
            panels.append(ElectricalPanel(
                equipment_id=eq_id,
                name=name,
                description=f"{rating}A {name}",
                location="Main Substation",
                building="Main Plant",
                floor="Ground Floor",
                zone="Electrical Room",
                manufacturer=random.choice(MANUFACTURERS["panel"]),
                installation_date=random_date(2018, 2021),
                status="running",
                criticality="critical",
                health_score=random_health(),
                panel_type=ptype,
                voltage_rating=433,
                current_rating=rating,
                short_circuit_rating=random.choice([25, 36, 50]),
                ip_rating="IP54",
                bus_bar_rating=rating,
                num_feeders=random.randint(8, 24),
                total_load_kw=rating * 0.433 * random.uniform(0.4, 0.8),
                total_current=rating * random.uniform(0.4, 0.8),
                voltage_r=random.uniform(225, 235),
                voltage_y=random.uniform(225, 235),
                voltage_b=random.uniform(225, 235),
                current_r=rating * random.uniform(0.3, 0.8) / 3,
                current_y=rating * random.uniform(0.3, 0.8) / 3,
                current_b=rating * random.uniform(0.3, 0.8) / 3,
                power_factor=random.uniform(0.85, 0.98),
                frequency=random.uniform(49.8, 50.2),
                total_energy_kwh=random.uniform(100000, 1000000),
                running_hours=random.uniform(20000, 50000),
            ))

        # Distribution boards
        for i in range(1, 35):
            building = random.choice(BUILDINGS)
            floor = random.choice(FLOORS)
            panels.append(ElectricalPanel(
                equipment_id=f"DB-{building[:2].upper()}-{floor[0]}{i:02d}",
                name=f"Distribution Board {i} - {building} {floor}",
                description=f"Distribution Board for {building} {floor}",
                location=f"{building} - {floor}",
                building=building,
                floor=floor,
                zone=random.choice(ZONES),
                manufacturer=random.choice(MANUFACTURERS["panel"]),
                installation_date=random_date(2019, 2023),
                status=random_status(),
                criticality=random.choice(["medium", "low"]),
                health_score=random_health(),
                panel_type="db",
                voltage_rating=433,
                current_rating=random.choice([100, 200, 250, 400, 630]),
                ip_rating=random.choice(["IP42", "IP54"]),
                num_feeders=random.randint(6, 18),
                total_load_kw=random.uniform(10, 100),
                power_factor=random.uniform(0.8, 0.95),
                total_energy_kwh=random.uniform(1000, 50000),
                running_hours=random.uniform(5000, 30000),
            ))

        # VFD Panels
        for i in range(1, 12):
            panels.append(ElectricalPanel(
                equipment_id=f"VFD-PNL-{i:02d}",
                name=f"VFD Panel {i}",
                description=f"Variable Frequency Drive Panel for {random.choice(['AHU', 'Pump', 'Fan', 'Compressor'])}",
                location=random.choice(BUILDINGS),
                building=random.choice(BUILDINGS),
                floor=random.choice(FLOORS),
                zone=random.choice(ZONES),
                manufacturer=random.choice(MANUFACTURERS["panel"]),
                installation_date=random_date(2020, 2024),
                status=random_status(),
                criticality="medium",
                health_score=random_health(),
                panel_type="vfd_panel",
                voltage_rating=433,
                current_rating=random.choice([50, 100, 150, 200]),
                ip_rating="IP54",
                running_hours=random.uniform(2000, 20000),
            ))

        ElectricalPanel.objects.bulk_create(panels)
        self.stdout.write(f"    Created {len(panels)} electrical panels")

    def create_ups_systems(self):
        """Create UPS records."""
        self.stdout.write("  Creating UPS systems...")
        ups_list = []

        # Large UPS systems
        large_capacities = [100, 120, 160, 200, 250, 300, 400, 500]
        for i, capacity in enumerate(large_capacities, 1):
            ups_list.append(UPS(
                equipment_id=f"UPS-L-{i:02d}",
                name=f"UPS {i} - {random.choice(['Data Center', 'Server Room', 'Critical Load'])}",
                description=f"{capacity} kVA Online UPS System",
                location=random.choice(BUILDINGS),
                building=random.choice(BUILDINGS),
                floor=random.choice(FLOORS),
                zone="UPS Room",
                manufacturer=random.choice(MANUFACTURERS["ups"]),
                model_number=f"UPS-{capacity}KVA",
                installation_date=random_date(2019, 2023),
                status=random_status(),
                criticality="critical",
                health_score=random_health(),
                ups_type="online",
                capacity_kva=capacity,
                input_voltage=433,
                output_voltage=433,
                battery_type="VRLA",
                battery_count=random.choice([32, 40, 48, 64]),
                backup_time_minutes=random.choice([10, 15, 20, 30]),
                load_percent=random.uniform(40, 80),
                battery_percent=random.uniform(85, 100),
                input_voltage_actual=random.uniform(415, 445),
                output_voltage_actual=random.uniform(225, 235),
                battery_voltage=random.uniform(380, 420),
                temperature=random.uniform(25, 35),
                on_battery=random.random() < 0.05,
                battery_health_percent=random.randint(75, 100),
                running_hours=random.uniform(10000, 40000),
            ))

        # Small UPS systems
        for i in range(1, 18):
            capacity = random.choice([3, 5, 6, 10, 15, 20])
            ups_list.append(UPS(
                equipment_id=f"UPS-S-{i:02d}",
                name=f"Small UPS {i}",
                description=f"{capacity} kVA UPS for {random.choice(['Workstations', 'Network Equipment', 'Control Systems'])}",
                location=random.choice(BUILDINGS),
                building=random.choice(BUILDINGS),
                floor=random.choice(FLOORS),
                zone=random.choice(ZONES),
                manufacturer=random.choice(MANUFACTURERS["ups"]),
                installation_date=random_date(2020, 2024),
                status=random_status(),
                criticality=random.choice(["medium", "low"]),
                health_score=random_health(),
                ups_type=random.choice(["online", "line_interactive"]),
                capacity_kva=capacity,
                input_voltage=230,
                output_voltage=230,
                battery_type="VRLA",
                battery_count=random.choice([2, 4, 6, 8]),
                backup_time_minutes=random.choice([5, 10, 15]),
                load_percent=random.uniform(30, 70),
                battery_percent=random.uniform(80, 100),
                running_hours=random.uniform(2000, 15000),
            ))

        UPS.objects.bulk_create(ups_list)
        self.stdout.write(f"    Created {len(ups_list)} UPS systems")

    def create_chillers(self):
        """Create chiller records."""
        self.stdout.write("  Creating chillers...")
        chillers = []

        capacities = [150, 200, 250, 300, 350, 400, 500, 600]
        for i, capacity in enumerate(capacities, 1):
            chiller_type = random.choice(["water_cooled", "air_cooled", "screw", "centrifugal"])
            chillers.append(Chiller(
                equipment_id=f"CH-{i:02d}",
                name=f"Chiller {i}",
                description=f"{capacity} TR {chiller_type.replace('_', ' ').title()} Chiller",
                location="Chiller Plant Room",
                building="Utility Building",
                floor="Ground Floor",
                zone="HVAC Plant Room",
                manufacturer=random.choice(MANUFACTURERS["chiller"]),
                model_number=f"CH-{capacity}TR",
                installation_date=random_date(2018, 2022),
                status=random_status(),
                criticality="critical",
                health_score=random_health(),
                chiller_type=chiller_type,
                capacity_tr=capacity,
                capacity_kw=capacity * 3.517,
                refrigerant_type=random.choice(["R134a", "R410A", "R407C"]),
                refrigerant_charge=random.uniform(50, 150),
                compressor_type=random.choice(["Screw", "Scroll", "Centrifugal"]),
                num_compressors=random.choice([1, 2]),
                cop_rating=random.uniform(4.5, 6.5),
                load_percent=random.uniform(40, 90),
                chilled_water_supply_temp=random.uniform(6, 8),
                chilled_water_return_temp=random.uniform(12, 14),
                condenser_water_in_temp=random.uniform(28, 32),
                condenser_water_out_temp=random.uniform(34, 38),
                evaporator_pressure=random.uniform(3, 5),
                condenser_pressure=random.uniform(10, 15),
                compressor_current=random.uniform(100, 300),
                power_consumption_kw=capacity * 3.517 / random.uniform(4.5, 6.0),
                current_cop=random.uniform(4.0, 6.0),
                running_hours=random.uniform(10000, 50000),
            ))

        # Smaller package chillers
        for i in range(9, 13):
            capacity = random.choice([20, 30, 40, 50])
            chillers.append(Chiller(
                equipment_id=f"CH-PKG-{i:02d}",
                name=f"Package Chiller {i}",
                description=f"{capacity} TR Package Air Cooled Chiller",
                location=random.choice(BUILDINGS),
                building=random.choice(BUILDINGS),
                floor="Terrace",
                zone="HVAC",
                manufacturer=random.choice(MANUFACTURERS["chiller"]),
                installation_date=random_date(2020, 2024),
                status=random_status(),
                criticality="medium",
                health_score=random_health(),
                chiller_type="air_cooled",
                capacity_tr=capacity,
                capacity_kw=capacity * 3.517,
                refrigerant_type="R410A",
                num_compressors=2,
                load_percent=random.uniform(30, 80),
                power_consumption_kw=capacity * 3.517 / random.uniform(3.5, 5.0),
                running_hours=random.uniform(5000, 25000),
            ))

        Chiller.objects.bulk_create(chillers)
        self.stdout.write(f"    Created {len(chillers)} chillers")

    def create_ahus(self):
        """Create AHU records."""
        self.stdout.write("  Creating AHUs...")
        ahus = []

        for i in range(1, 45):
            building = random.choice(BUILDINGS)
            floor = random.choice(FLOORS)
            ahu_type = random.choice(["fresh_air", "return_air", "mixed_air"])
            capacity = random.choice([5000, 8000, 10000, 12000, 15000, 20000, 25000])

            ahus.append(AHU(
                equipment_id=f"AHU-{building[:2].upper()}-{floor[0]}{i:02d}",
                name=f"AHU {i} - {building} {floor}",
                description=f"{capacity} CFM {ahu_type.replace('_', ' ').title()} AHU",
                location=f"{building} - {floor}",
                building=building,
                floor=floor,
                zone=random.choice(ZONES),
                manufacturer=random.choice(MANUFACTURERS["ahu"]),
                model_number=f"AHU-{capacity}CFM",
                installation_date=random_date(2019, 2023),
                status=random_status(),
                criticality=random.choice(["high", "medium"]),
                health_score=random_health(),
                ahu_type=ahu_type,
                capacity_cfm=capacity,
                capacity_cmh=capacity * 1.699,
                cooling_capacity_kw=capacity * 0.003,
                fan_motor_kw=random.choice([5.5, 7.5, 11, 15, 18.5, 22]),
                filter_type=random.choice(["MERV 8", "MERV 13", "HEPA"]),
                has_vfd=random.random() > 0.3,
                supply_air_temp=random.uniform(14, 18),
                return_air_temp=random.uniform(24, 28),
                supply_air_humidity=random.uniform(50, 65),
                return_air_humidity=random.uniform(55, 70),
                fan_speed_percent=random.uniform(50, 100),
                damper_position_percent=random.uniform(20, 80),
                filter_dp=random.uniform(50, 250),
                chilled_water_valve_percent=random.uniform(30, 80),
                power_consumption_kw=random.uniform(3, 20),
                running_hours=random.uniform(5000, 40000),
            ))

        AHU.objects.bulk_create(ahus)
        self.stdout.write(f"    Created {len(ahus)} AHUs")

    def create_cooling_towers(self):
        """Create cooling tower records."""
        self.stdout.write("  Creating cooling towers...")
        towers = []

        capacities = [200, 250, 300, 350, 400, 500, 600, 750]
        for i, capacity in enumerate(capacities, 1):
            towers.append(CoolingTower(
                equipment_id=f"CT-{i:02d}",
                name=f"Cooling Tower {i}",
                description=f"{capacity} TR Induced Draft Cooling Tower",
                location="Cooling Tower Yard",
                building="Utility Building",
                floor="Terrace",
                zone="CT Yard",
                manufacturer=random.choice(["Paharpur", "Baltimore Aircoil", "Marley", "SPX"]),
                model_number=f"CT-{capacity}TR",
                installation_date=random_date(2018, 2022),
                status=random_status(),
                criticality="critical",
                health_score=random_health(),
                tower_type=random.choice(["induced_draft", "cross_flow", "counter_flow"]),
                capacity_tr=capacity,
                water_flow_rate=capacity * 3,
                fan_motor_kw=random.choice([15, 18.5, 22, 30, 37]),
                num_cells=random.choice([1, 2]),
                inlet_water_temp=random.uniform(34, 40),
                outlet_water_temp=random.uniform(28, 32),
                wet_bulb_temp=random.uniform(24, 28),
                fan_speed_percent=random.uniform(60, 100),
                water_level_percent=random.uniform(70, 95),
                conductivity=random.uniform(500, 1500),
                power_consumption_kw=random.uniform(10, 35),
                running_hours=random.uniform(15000, 50000),
            ))

        CoolingTower.objects.bulk_create(towers)
        self.stdout.write(f"    Created {len(towers)} cooling towers")

    def create_pumps(self):
        """Create pump records."""
        self.stdout.write("  Creating pumps...")
        pumps = []

        pump_configs = [
            ("chw_primary", "CHWP", "Chilled Water Primary", 8),
            ("chw_secondary", "CHWS", "Chilled Water Secondary", 10),
            ("condenser", "CDWP", "Condenser Water", 8),
            ("hot_water", "HTWP", "Hot Water", 4),
            ("fire", "FIRE", "Fire Pump", 3),
            ("sump", "SUMP", "Sump Pump", 6),
            ("booster", "BSTR", "Booster Pump", 8),
            ("transfer", "XFER", "Transfer Pump", 5),
            ("dosing", "DOSE", "Dosing Pump", 8),
        ]

        for ptype, prefix, pname, count in pump_configs:
            for i in range(1, count + 1):
                flow = random.choice([20, 30, 50, 80, 100, 150, 200, 300])
                head = random.choice([20, 30, 40, 50, 60])
                motor_kw = flow * head * 9.81 / (3600 * 1000 * 0.7)  # Rough calculation

                pumps.append(Pump(
                    equipment_id=f"PMP-{prefix}-{i:02d}",
                    name=f"{pname} Pump {i}",
                    description=f"{flow} m³/hr @ {head}m head",
                    location="Pump Room" if ptype != "sump" else random.choice(BUILDINGS),
                    building=random.choice(BUILDINGS),
                    floor="Basement" if ptype == "sump" else "Ground Floor",
                    zone="Pump Room",
                    manufacturer=random.choice(MANUFACTURERS["pump"]),
                    model_number=f"PMP-{int(flow)}-{int(head)}",
                    installation_date=random_date(2018, 2023),
                    status=random_status(),
                    criticality="critical" if ptype in ["chw_primary", "condenser", "fire"] else "high",
                    health_score=random_health(),
                    pump_type=ptype,
                    flow_rate=flow,
                    head=head,
                    motor_kw=round(motor_kw * 1.5, 1),  # With safety margin
                    rpm=random.choice([1450, 1750, 2900]),
                    has_vfd=random.random() > 0.4,
                    flow_rate_actual=flow * random.uniform(0.7, 1.0) if random.random() > 0.3 else None,
                    discharge_pressure=head * 0.0981 * random.uniform(0.9, 1.1),
                    suction_pressure=random.uniform(0.5, 2.0),
                    motor_current=motor_kw * 1.5 / 0.433 / 0.85 * random.uniform(0.5, 0.9),
                    speed_percent=random.uniform(60, 100),
                    vibration=random.uniform(0.5, 4.5),
                    bearing_temperature=random.uniform(35, 65),
                    power_consumption_kw=motor_kw * 1.5 * random.uniform(0.5, 0.9),
                    running_hours=random.uniform(5000, 40000),
                ))

        Pump.objects.bulk_create(pumps)
        self.stdout.write(f"    Created {len(pumps)} pumps")

    def create_compressors(self):
        """Create compressor records."""
        self.stdout.write("  Creating compressors...")
        compressors = []

        for i in range(1, 16):
            capacity = random.choice([100, 150, 200, 300, 500, 750, 1000])
            compressors.append(Compressor(
                equipment_id=f"COMP-{i:02d}",
                name=f"Air Compressor {i}",
                description=f"{capacity} CFM Screw Compressor",
                location="Compressor Room",
                building=random.choice(BUILDINGS),
                floor="Ground Floor",
                zone="Utility Area",
                manufacturer=random.choice(MANUFACTURERS["compressor"]),
                model_number=f"SC-{capacity}",
                installation_date=random_date(2019, 2023),
                status=random_status(),
                criticality="high",
                health_score=random_health(),
                compressor_type=random.choice(["screw", "reciprocating"]),
                capacity_cfm=capacity,
                capacity_m3h=capacity * 1.699,
                working_pressure=random.choice([7, 8, 10, 12]),
                motor_kw=capacity * 0.15,  # Rough estimate
                has_vfd=random.random() > 0.5,
                tank_capacity=random.choice([500, 1000, 2000]),
                discharge_pressure=random.uniform(6.5, 10),
                inlet_pressure=random.uniform(0.9, 1.1),
                discharge_temperature=random.uniform(60, 90),
                oil_temperature=random.uniform(40, 70),
                oil_pressure=random.uniform(2, 4),
                motor_current=capacity * 0.15 / 0.433 / 0.85 * random.uniform(0.5, 0.9),
                load_percent=random.uniform(40, 90),
                power_consumption_kw=capacity * 0.15 * random.uniform(0.5, 0.9),
                running_hours=random.uniform(5000, 30000),
            ))

        Compressor.objects.bulk_create(compressors)
        self.stdout.write(f"    Created {len(compressors)} compressors")

    def create_motors(self):
        """Create motor records."""
        self.stdout.write("  Creating motors...")
        motors = []

        for i in range(1, 85):
            power = random.choice([0.75, 1.1, 1.5, 2.2, 3.7, 5.5, 7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75])
            motors.append(Motor(
                equipment_id=f"MTR-{i:03d}",
                name=f"Motor {i}",
                description=f"{power} kW {random.choice(['Conveyor', 'Fan', 'Blower', 'Mixer', 'Agitator', 'Crusher'])} Motor",
                location=random.choice(BUILDINGS),
                building=random.choice(BUILDINGS),
                floor=random.choice(FLOORS),
                zone=random.choice(ZONES),
                manufacturer=random.choice(MANUFACTURERS["motor"]),
                model_number=f"MTR-{int(power*10)}",
                installation_date=random_date(2018, 2024),
                status=random_status(),
                criticality=random.choice(["high", "medium", "low"]),
                health_score=random_health(),
                motor_type="induction",
                power_kw=power,
                voltage=415,
                current_rating=power / 0.415 / 0.85 / 1.732,
                rpm=random.choice([960, 1440, 2880]),
                efficiency_class=random.choice(["IE2", "IE3", "IE4"]),
                frame_size=f"{int(power*10)}M",
                duty_cycle="S1",
                has_vfd=random.random() > 0.6,
                current_r=power / 0.415 / 0.85 / 1.732 * random.uniform(0.5, 0.95),
                current_y=power / 0.415 / 0.85 / 1.732 * random.uniform(0.5, 0.95),
                current_b=power / 0.415 / 0.85 / 1.732 * random.uniform(0.5, 0.95),
                voltage_actual=random.uniform(400, 430),
                speed_rpm=random.choice([960, 1440, 2880]) * random.uniform(0.9, 1.0),
                winding_temperature=random.uniform(40, 80),
                bearing_temperature=random.uniform(35, 65),
                vibration=random.uniform(0.5, 5.0),
                power_consumption_kw=power * random.uniform(0.5, 0.95),
                power_factor=random.uniform(0.75, 0.92),
                running_hours=random.uniform(2000, 30000),
            ))

        Motor.objects.bulk_create(motors)
        self.stdout.write(f"    Created {len(motors)} motors")

    def create_energy_meters(self):
        """Create energy meter records."""
        self.stdout.write("  Creating energy meters...")
        meters = []

        # Main incomer meters
        for i in range(1, 6):
            meters.append(EnergyMeter(
                equipment_id=f"EM-INC-{i:02d}",
                name=f"Main Incomer Meter {i}",
                description=f"Main Grid Incomer Energy Meter",
                location="Main Substation",
                building="Main Plant",
                floor="Ground Floor",
                zone="Substation",
                manufacturer=random.choice(MANUFACTURERS["meter"]),
                installation_date=random_date(2018, 2021),
                status="running",
                criticality="critical",
                health_score=100,
                meter_type="main_incomer",
                ct_ratio="1000/5",
                pt_ratio="11000/110",
                accuracy_class="0.2S",
                voltage_r=random.uniform(225, 235),
                voltage_y=random.uniform(225, 235),
                voltage_b=random.uniform(225, 235),
                voltage_ry=random.uniform(395, 420),
                voltage_yb=random.uniform(395, 420),
                voltage_br=random.uniform(395, 420),
                current_r=random.uniform(200, 800),
                current_y=random.uniform(200, 800),
                current_b=random.uniform(200, 800),
                power_kw=random.uniform(500, 2000),
                power_kva=random.uniform(600, 2200),
                power_kvar=random.uniform(100, 400),
                power_factor=random.uniform(0.88, 0.98),
                frequency=random.uniform(49.9, 50.1),
                total_kwh=random.uniform(1000000, 10000000),
                total_kvah=random.uniform(1100000, 11000000),
                max_demand_kw=random.uniform(1500, 3000),
                max_demand_kva=random.uniform(1700, 3300),
                running_hours=random.uniform(30000, 50000),
            ))

        # Sub meters and feeder meters
        for i in range(1, 100):
            building = random.choice(BUILDINGS)
            meter_type = random.choice(["sub_meter", "feeder", "dg_meter"])
            meters.append(EnergyMeter(
                equipment_id=f"EM-{meter_type.upper()[:3]}-{i:03d}",
                name=f"{meter_type.replace('_', ' ').title()} {i} - {building}",
                description=f"Energy Meter for {building}",
                location=building,
                building=building,
                floor=random.choice(FLOORS),
                zone=random.choice(ZONES),
                manufacturer=random.choice(MANUFACTURERS["meter"]),
                installation_date=random_date(2019, 2024),
                status=random_status(),
                criticality=random.choice(["medium", "low"]),
                health_score=random_health(),
                meter_type=meter_type,
                ct_ratio=random.choice(["100/5", "200/5", "400/5", "500/5"]),
                accuracy_class=random.choice(["0.5S", "1.0"]),
                power_kw=random.uniform(10, 200),
                power_factor=random.uniform(0.8, 0.95),
                total_kwh=random.uniform(10000, 500000),
                running_hours=random.uniform(5000, 30000),
            ))

        EnergyMeter.objects.bulk_create(meters)
        self.stdout.write(f"    Created {len(meters)} energy meters")

    def create_alerts(self):
        """Create alert records."""
        self.stdout.write("  Creating alerts...")
        alerts = []

        alert_templates = [
            ("Transformer", "TR-MAIN-01", "Main Transformer 1", "threshold", "high", "Oil temperature high", "oil_temperature", 82, 75, "°C"),
            ("Transformer", "TR-DRY-05", "Dry Type Transformer 5", "threshold", "medium", "Winding temperature elevated", "winding_temperature", 72, 70, "°C"),
            ("DieselGenerator", "DG-02", "DG Set 2", "fault", "critical", "Engine failed to start", None, None, None, None),
            ("DieselGenerator", "DG-05", "DG Set 5", "threshold", "medium", "Fuel level low", "fuel_level_percent", 25, 30, "%"),
            ("Chiller", "CH-03", "Chiller 3", "threshold", "high", "Compressor discharge pressure high", "condenser_pressure", 16.5, 15, "bar"),
            ("Chiller", "CH-01", "Chiller 1", "maintenance", "medium", "Preventive maintenance due", None, None, None, None),
            ("AHU", "AHU-MA-G12", "AHU 12 - Main Plant Ground Floor", "threshold", "medium", "Filter differential pressure high", "filter_dp", 280, 250, "Pa"),
            ("Pump", "PMP-CHW-02", "Chilled Water Primary Pump 2", "threshold", "high", "Bearing temperature high", "bearing_temperature", 72, 65, "°C"),
            ("Pump", "PMP-CON-03", "Condenser Water Pump 3", "threshold", "medium", "Vibration elevated", "vibration", 5.2, 4.5, "mm/s"),
            ("Motor", "MTR-045", "Motor 45", "fault", "high", "Overcurrent trip", "current_r", 125, 100, "A"),
            ("Compressor", "COMP-03", "Air Compressor 3", "threshold", "medium", "Discharge temperature high", "discharge_temperature", 95, 90, "°C"),
            ("UPS", "UPS-L-02", "UPS 2 - Data Center", "threshold", "critical", "Battery backup low", "battery_percent", 45, 50, "%"),
            ("UPS", "UPS-L-04", "UPS 4 - Server Room", "communication", "high", "Communication loss with BMS", None, None, None, None),
            ("CoolingTower", "CT-02", "Cooling Tower 2", "threshold", "medium", "Water conductivity high", "conductivity", 1800, 1500, "µS/cm"),
            ("ElectricalPanel", "APFC-1", "Auto Power Factor Panel 1", "threshold", "medium", "Power factor low", "power_factor", 0.82, 0.85, None),
        ]

        for eq_type, eq_id, eq_name, alert_type, severity, message, param, value, threshold, unit in alert_templates:
            # Create multiple instances of each alert type
            for _ in range(random.randint(1, 5)):
                alerts.append(Alert(
                    equipment_type=eq_type,
                    equipment_id=eq_id,
                    equipment_name=eq_name,
                    severity=severity,
                    alert_type=alert_type,
                    message=message,
                    parameter=param or "",
                    value=value * random.uniform(0.95, 1.05) if value else None,
                    threshold=threshold,
                    unit=unit or "",
                    triggered_at=timezone.now() - timedelta(hours=random.randint(0, 720)),
                    acknowledged=random.random() > 0.4,
                    resolved=random.random() > 0.5,
                ))

        Alert.objects.bulk_create(alerts)
        self.stdout.write(f"    Created {len(alerts)} alerts")

    def create_maintenance_records(self):
        """Create maintenance records."""
        self.stdout.write("  Creating maintenance records...")
        records = []

        equipment_list = [
            ("Transformer", "TR-MAIN-01", "Main Transformer 1"),
            ("Transformer", "TR-DRY-05", "Dry Type Transformer 5"),
            ("DieselGenerator", "DG-01", "DG Set 1"),
            ("DieselGenerator", "DG-02", "DG Set 2"),
            ("Chiller", "CH-01", "Chiller 1"),
            ("Chiller", "CH-02", "Chiller 2"),
            ("AHU", "AHU-MA-G01", "AHU 1 - Main Plant"),
            ("Pump", "PMP-CHW-01", "Chilled Water Primary Pump 1"),
            ("Pump", "PMP-CON-01", "Condenser Water Pump 1"),
            ("Compressor", "COMP-01", "Air Compressor 1"),
            ("Motor", "MTR-001", "Motor 1"),
            ("UPS", "UPS-L-01", "UPS 1 - Data Center"),
            ("CoolingTower", "CT-01", "Cooling Tower 1"),
        ]

        maintenance_types = ["preventive", "corrective", "breakdown", "inspection"]
        descriptions = [
            "Annual preventive maintenance",
            "Quarterly inspection",
            "Bearing replacement",
            "Filter cleaning/replacement",
            "Oil change and analysis",
            "Vibration analysis",
            "Thermography inspection",
            "Electrical testing",
            "Control system calibration",
            "Safety device testing",
            "Performance optimization",
            "Emergency repair",
        ]

        for eq_type, eq_id, eq_name in equipment_list:
            # Create multiple maintenance records for each equipment
            for _ in range(random.randint(5, 20)):
                mtype = random.choice(maintenance_types)
                records.append(MaintenanceRecord(
                    equipment_type=eq_type,
                    equipment_id=eq_id,
                    equipment_name=eq_name,
                    maintenance_type=mtype,
                    description=random.choice(descriptions),
                    work_done=f"Completed {mtype} maintenance activities",
                    parts_replaced="Various consumables" if random.random() > 0.5 else "",
                    cost=random.uniform(1000, 50000) if random.random() > 0.3 else None,
                    technician=f"Tech-{random.randint(1, 20):02d}",
                    vendor=random.choice(["Internal", "ABC Services", "XYZ Maintenance", "OEM Support"]),
                    scheduled_date=timezone.now() - timedelta(days=random.randint(0, 365)),
                    completed_at=timezone.now() - timedelta(days=random.randint(0, 365)),
                    downtime_hours=random.uniform(0.5, 24) if mtype != "inspection" else 0,
                ))

        MaintenanceRecord.objects.bulk_create(records)
        self.stdout.write(f"    Created {len(records)} maintenance records")

    def print_summary(self):
        """Print summary of created records."""
        self.stdout.write("\n" + "=" * 50)
        self.stdout.write("DATABASE SUMMARY")
        self.stdout.write("=" * 50)

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
            ("Alerts", Alert),
            ("Maintenance Records", MaintenanceRecord),
        ]

        total = 0
        for name, model in models:
            count = model.objects.count()
            total += count
            self.stdout.write(f"  {name}: {count}")

        self.stdout.write("=" * 50)
        self.stdout.write(f"  TOTAL RECORDS: {total}")
        self.stdout.write("=" * 50)
