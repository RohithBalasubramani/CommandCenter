"""
Industrial Energy Database Models

Contains models for all industrial equipment types:
- Transformers
- Diesel Generators (DG Sets)
- Electrical Panels (MCC, PCC, APFC, etc.)
- HVAC Equipment (AHUs, Chillers, Cooling Towers)
- Pumps and Motors
- Compressors
- Meters and Sensors
- And more...
"""

import uuid
from django.db import models
from django.utils import timezone


class BaseEquipment(models.Model):
    """Abstract base class for all equipment."""

    class Status(models.TextChoices):
        RUNNING = "running", "Running"
        STOPPED = "stopped", "Stopped"
        MAINTENANCE = "maintenance", "Under Maintenance"
        FAULT = "fault", "Fault"
        STANDBY = "standby", "Standby"
        OFFLINE = "offline", "Offline"

    class Criticality(models.TextChoices):
        CRITICAL = "critical", "Critical"
        HIGH = "high", "High"
        MEDIUM = "medium", "Medium"
        LOW = "low", "Low"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    equipment_id = models.CharField(max_length=50, unique=True, db_index=True)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    location = models.CharField(max_length=200)
    building = models.CharField(max_length=100, blank=True)
    floor = models.CharField(max_length=50, blank=True)
    zone = models.CharField(max_length=100, blank=True)

    manufacturer = models.CharField(max_length=200, blank=True)
    model_number = models.CharField(max_length=100, blank=True)
    serial_number = models.CharField(max_length=100, blank=True)
    installation_date = models.DateField(null=True, blank=True)
    warranty_expiry = models.DateField(null=True, blank=True)

    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.STOPPED
    )
    criticality = models.CharField(
        max_length=20, choices=Criticality.choices, default=Criticality.MEDIUM
    )
    health_score = models.IntegerField(default=100)  # 0-100

    last_maintenance = models.DateTimeField(null=True, blank=True)
    next_maintenance = models.DateTimeField(null=True, blank=True)
    running_hours = models.FloatField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True

    def __str__(self):
        return f"{self.equipment_id} - {self.name}"


# ============================================================
# Electrical Equipment
# ============================================================


class Transformer(BaseEquipment):
    """Power transformers (distribution, step-up, step-down)."""

    class TransformerType(models.TextChoices):
        DISTRIBUTION = "distribution", "Distribution Transformer"
        POWER = "power", "Power Transformer"
        DRY_TYPE = "dry_type", "Dry Type Transformer"
        OIL_FILLED = "oil_filled", "Oil Filled Transformer"
        AUTO = "auto", "Auto Transformer"

    transformer_type = models.CharField(
        max_length=20, choices=TransformerType.choices, default=TransformerType.DISTRIBUTION
    )
    capacity_kva = models.FloatField()  # kVA rating
    primary_voltage = models.FloatField()  # V
    secondary_voltage = models.FloatField()  # V
    vector_group = models.CharField(max_length=20, blank=True)  # e.g., Dyn11
    impedance_percent = models.FloatField(null=True, blank=True)
    cooling_type = models.CharField(max_length=50, blank=True)  # ONAN, ONAF, etc.

    # Real-time readings
    load_percent = models.FloatField(default=0)  # Current load %
    oil_temperature = models.FloatField(null=True, blank=True)  # °C
    winding_temperature = models.FloatField(null=True, blank=True)  # °C
    input_voltage = models.FloatField(null=True, blank=True)
    output_voltage = models.FloatField(null=True, blank=True)
    current_load_kw = models.FloatField(default=0)

    class Meta:
        verbose_name = "Transformer"
        verbose_name_plural = "Transformers"


class DieselGenerator(BaseEquipment):
    """Diesel Generator Sets (DG Sets)."""

    capacity_kva = models.FloatField()  # kVA rating
    capacity_kw = models.FloatField()  # kW rating
    voltage = models.FloatField()  # Output voltage
    frequency = models.FloatField(default=50)  # Hz
    power_factor = models.FloatField(default=0.8)
    phases = models.IntegerField(default=3)

    engine_make = models.CharField(max_length=100, blank=True)
    engine_model = models.CharField(max_length=100, blank=True)
    alternator_make = models.CharField(max_length=100, blank=True)
    fuel_tank_capacity = models.FloatField(null=True, blank=True)  # Liters

    # Real-time readings
    current_load_kw = models.FloatField(default=0)
    load_percent = models.FloatField(default=0)
    fuel_level_percent = models.FloatField(default=100)
    coolant_temperature = models.FloatField(null=True, blank=True)  # °C
    oil_pressure = models.FloatField(null=True, blank=True)  # bar
    battery_voltage = models.FloatField(null=True, blank=True)
    output_voltage = models.FloatField(null=True, blank=True)
    output_frequency = models.FloatField(null=True, blank=True)
    total_run_hours = models.FloatField(default=0)
    total_energy_kwh = models.FloatField(default=0)

    class Meta:
        verbose_name = "Diesel Generator"
        verbose_name_plural = "Diesel Generators"


class ElectricalPanel(BaseEquipment):
    """Electrical panels (MCC, PCC, APFC, DB, etc.)."""

    class PanelType(models.TextChoices):
        MCC = "mcc", "Motor Control Center"
        PCC = "pcc", "Power Control Center"
        APFC = "apfc", "Automatic Power Factor Controller"
        DB = "db", "Distribution Board"
        MLDB = "mldb", "Main LT Distribution Board"
        SMDB = "smdb", "Sub Main Distribution Board"
        PMCC = "pmcc", "Power & Motor Control Center"
        VFD_PANEL = "vfd_panel", "VFD Panel"
        PLC_PANEL = "plc_panel", "PLC Panel"
        ATS = "ats", "Auto Transfer Switch"
        CHANGEOVER = "changeover", "Changeover Panel"

    panel_type = models.CharField(
        max_length=20, choices=PanelType.choices, default=PanelType.DB
    )
    voltage_rating = models.FloatField()  # V
    current_rating = models.FloatField()  # A
    short_circuit_rating = models.FloatField(null=True, blank=True)  # kA
    ip_rating = models.CharField(max_length=10, blank=True)  # IP54, IP65, etc.
    bus_bar_rating = models.FloatField(null=True, blank=True)  # A
    num_feeders = models.IntegerField(default=0)

    # Real-time readings
    total_load_kw = models.FloatField(default=0)
    total_current = models.FloatField(default=0)
    voltage_r = models.FloatField(null=True, blank=True)
    voltage_y = models.FloatField(null=True, blank=True)
    voltage_b = models.FloatField(null=True, blank=True)
    current_r = models.FloatField(null=True, blank=True)
    current_y = models.FloatField(null=True, blank=True)
    current_b = models.FloatField(null=True, blank=True)
    power_factor = models.FloatField(null=True, blank=True)
    frequency = models.FloatField(null=True, blank=True)
    total_energy_kwh = models.FloatField(default=0)

    class Meta:
        verbose_name = "Electrical Panel"
        verbose_name_plural = "Electrical Panels"


class UPS(BaseEquipment):
    """Uninterruptible Power Supply systems."""

    class UPSType(models.TextChoices):
        ONLINE = "online", "Online Double Conversion"
        OFFLINE = "offline", "Offline/Standby"
        LINE_INTERACTIVE = "line_interactive", "Line Interactive"

    ups_type = models.CharField(
        max_length=20, choices=UPSType.choices, default=UPSType.ONLINE
    )
    capacity_kva = models.FloatField()
    input_voltage = models.FloatField()
    output_voltage = models.FloatField()
    battery_type = models.CharField(max_length=50, blank=True)
    battery_count = models.IntegerField(default=0)
    backup_time_minutes = models.IntegerField(null=True, blank=True)

    # Real-time readings
    load_percent = models.FloatField(default=0)
    battery_percent = models.FloatField(default=100)
    input_voltage_actual = models.FloatField(null=True, blank=True)
    output_voltage_actual = models.FloatField(null=True, blank=True)
    battery_voltage = models.FloatField(null=True, blank=True)
    temperature = models.FloatField(null=True, blank=True)
    on_battery = models.BooleanField(default=False)
    battery_health_percent = models.IntegerField(default=100)

    class Meta:
        verbose_name = "UPS"
        verbose_name_plural = "UPS Systems"


# ============================================================
# HVAC Equipment
# ============================================================


class Chiller(BaseEquipment):
    """Chillers (air-cooled, water-cooled)."""

    class ChillerType(models.TextChoices):
        AIR_COOLED = "air_cooled", "Air Cooled"
        WATER_COOLED = "water_cooled", "Water Cooled"
        ABSORPTION = "absorption", "Absorption Chiller"
        CENTRIFUGAL = "centrifugal", "Centrifugal"
        SCREW = "screw", "Screw Chiller"
        SCROLL = "scroll", "Scroll Chiller"

    chiller_type = models.CharField(
        max_length=20, choices=ChillerType.choices, default=ChillerType.AIR_COOLED
    )
    capacity_tr = models.FloatField()  # Tons of Refrigeration
    capacity_kw = models.FloatField()  # Cooling capacity kW
    refrigerant_type = models.CharField(max_length=20, blank=True)  # R134a, R410A, etc.
    refrigerant_charge = models.FloatField(null=True, blank=True)  # kg
    compressor_type = models.CharField(max_length=50, blank=True)
    num_compressors = models.IntegerField(default=1)
    cop_rating = models.FloatField(null=True, blank=True)  # Coefficient of Performance

    # Real-time readings
    load_percent = models.FloatField(default=0)
    chilled_water_supply_temp = models.FloatField(null=True, blank=True)  # °C
    chilled_water_return_temp = models.FloatField(null=True, blank=True)  # °C
    condenser_water_in_temp = models.FloatField(null=True, blank=True)  # °C
    condenser_water_out_temp = models.FloatField(null=True, blank=True)  # °C
    evaporator_pressure = models.FloatField(null=True, blank=True)  # bar
    condenser_pressure = models.FloatField(null=True, blank=True)  # bar
    compressor_current = models.FloatField(null=True, blank=True)  # A
    power_consumption_kw = models.FloatField(default=0)
    current_cop = models.FloatField(null=True, blank=True)

    class Meta:
        verbose_name = "Chiller"
        verbose_name_plural = "Chillers"


class AHU(BaseEquipment):
    """Air Handling Units."""

    class AHUType(models.TextChoices):
        FRESH_AIR = "fresh_air", "Fresh Air AHU"
        RETURN_AIR = "return_air", "Return Air AHU"
        MIXED_AIR = "mixed_air", "Mixed Air AHU"
        HEAT_RECOVERY = "heat_recovery", "Heat Recovery AHU"
        ROOFTOP = "rooftop", "Rooftop Unit"

    ahu_type = models.CharField(
        max_length=20, choices=AHUType.choices, default=AHUType.MIXED_AIR
    )
    capacity_cfm = models.FloatField()  # Cubic Feet per Minute
    capacity_cmh = models.FloatField()  # Cubic Meters per Hour
    cooling_capacity_kw = models.FloatField(null=True, blank=True)
    heating_capacity_kw = models.FloatField(null=True, blank=True)
    fan_motor_kw = models.FloatField(null=True, blank=True)
    filter_type = models.CharField(max_length=50, blank=True)
    has_vfd = models.BooleanField(default=False)

    # Real-time readings
    supply_air_temp = models.FloatField(null=True, blank=True)  # °C
    return_air_temp = models.FloatField(null=True, blank=True)  # °C
    supply_air_humidity = models.FloatField(null=True, blank=True)  # %RH
    return_air_humidity = models.FloatField(null=True, blank=True)  # %RH
    fan_speed_percent = models.FloatField(default=0)
    damper_position_percent = models.FloatField(default=0)
    filter_dp = models.FloatField(null=True, blank=True)  # Filter differential pressure (Pa)
    chilled_water_valve_percent = models.FloatField(null=True, blank=True)
    power_consumption_kw = models.FloatField(default=0)

    class Meta:
        verbose_name = "AHU"
        verbose_name_plural = "AHUs"


class CoolingTower(BaseEquipment):
    """Cooling Towers."""

    class TowerType(models.TextChoices):
        INDUCED_DRAFT = "induced_draft", "Induced Draft"
        FORCED_DRAFT = "forced_draft", "Forced Draft"
        CROSS_FLOW = "cross_flow", "Cross Flow"
        COUNTER_FLOW = "counter_flow", "Counter Flow"

    tower_type = models.CharField(
        max_length=20, choices=TowerType.choices, default=TowerType.INDUCED_DRAFT
    )
    capacity_tr = models.FloatField()  # Tons of Refrigeration
    water_flow_rate = models.FloatField()  # m³/hr
    fan_motor_kw = models.FloatField(null=True, blank=True)
    num_cells = models.IntegerField(default=1)

    # Real-time readings
    inlet_water_temp = models.FloatField(null=True, blank=True)  # °C
    outlet_water_temp = models.FloatField(null=True, blank=True)  # °C
    wet_bulb_temp = models.FloatField(null=True, blank=True)  # °C
    fan_speed_percent = models.FloatField(default=0)
    water_level_percent = models.FloatField(null=True, blank=True)
    conductivity = models.FloatField(null=True, blank=True)  # µS/cm
    power_consumption_kw = models.FloatField(default=0)

    class Meta:
        verbose_name = "Cooling Tower"
        verbose_name_plural = "Cooling Towers"


# ============================================================
# Mechanical Equipment
# ============================================================


class Pump(BaseEquipment):
    """Pumps (water, chilled water, condenser, etc.)."""

    class PumpType(models.TextChoices):
        CHILLED_WATER_PRIMARY = "chw_primary", "Chilled Water Primary"
        CHILLED_WATER_SECONDARY = "chw_secondary", "Chilled Water Secondary"
        CONDENSER_WATER = "condenser", "Condenser Water"
        HOT_WATER = "hot_water", "Hot Water"
        FIRE_PUMP = "fire", "Fire Pump"
        SUMP_PUMP = "sump", "Sump Pump"
        BOOSTER_PUMP = "booster", "Booster Pump"
        TRANSFER_PUMP = "transfer", "Transfer Pump"
        DOSING_PUMP = "dosing", "Dosing Pump"

    pump_type = models.CharField(
        max_length=20, choices=PumpType.choices, default=PumpType.CHILLED_WATER_PRIMARY
    )
    flow_rate = models.FloatField()  # m³/hr
    head = models.FloatField()  # meters
    motor_kw = models.FloatField()
    rpm = models.IntegerField(null=True, blank=True)
    impeller_diameter = models.FloatField(null=True, blank=True)  # mm
    has_vfd = models.BooleanField(default=False)

    # Real-time readings
    flow_rate_actual = models.FloatField(null=True, blank=True)  # m³/hr
    discharge_pressure = models.FloatField(null=True, blank=True)  # bar
    suction_pressure = models.FloatField(null=True, blank=True)  # bar
    motor_current = models.FloatField(null=True, blank=True)  # A
    speed_percent = models.FloatField(default=0)
    vibration = models.FloatField(null=True, blank=True)  # mm/s
    bearing_temperature = models.FloatField(null=True, blank=True)  # °C
    power_consumption_kw = models.FloatField(default=0)

    class Meta:
        verbose_name = "Pump"
        verbose_name_plural = "Pumps"


class Compressor(BaseEquipment):
    """Air Compressors."""

    class CompressorType(models.TextChoices):
        SCREW = "screw", "Screw Compressor"
        RECIPROCATING = "reciprocating", "Reciprocating"
        CENTRIFUGAL = "centrifugal", "Centrifugal"
        SCROLL = "scroll", "Scroll"
        ROTARY_VANE = "rotary_vane", "Rotary Vane"

    compressor_type = models.CharField(
        max_length=20, choices=CompressorType.choices, default=CompressorType.SCREW
    )
    capacity_cfm = models.FloatField()  # CFM
    capacity_m3h = models.FloatField()  # m³/hr
    working_pressure = models.FloatField()  # bar
    motor_kw = models.FloatField()
    has_vfd = models.BooleanField(default=False)
    tank_capacity = models.FloatField(null=True, blank=True)  # liters

    # Real-time readings
    discharge_pressure = models.FloatField(null=True, blank=True)  # bar
    inlet_pressure = models.FloatField(null=True, blank=True)  # bar
    discharge_temperature = models.FloatField(null=True, blank=True)  # °C
    oil_temperature = models.FloatField(null=True, blank=True)  # °C
    oil_pressure = models.FloatField(null=True, blank=True)  # bar
    motor_current = models.FloatField(null=True, blank=True)  # A
    load_percent = models.FloatField(default=0)
    power_consumption_kw = models.FloatField(default=0)

    class Meta:
        verbose_name = "Compressor"
        verbose_name_plural = "Compressors"


class Motor(BaseEquipment):
    """Electric Motors (standalone)."""

    class MotorType(models.TextChoices):
        INDUCTION = "induction", "Induction Motor"
        SYNCHRONOUS = "synchronous", "Synchronous Motor"
        DC = "dc", "DC Motor"
        SERVO = "servo", "Servo Motor"
        STEPPER = "stepper", "Stepper Motor"

    motor_type = models.CharField(
        max_length=20, choices=MotorType.choices, default=MotorType.INDUCTION
    )
    power_kw = models.FloatField()
    voltage = models.FloatField()
    current_rating = models.FloatField()  # A
    rpm = models.IntegerField()
    efficiency_class = models.CharField(max_length=10, blank=True)  # IE1, IE2, IE3, IE4
    frame_size = models.CharField(max_length=20, blank=True)
    duty_cycle = models.CharField(max_length=10, blank=True)  # S1, S2, etc.
    has_vfd = models.BooleanField(default=False)

    # Real-time readings
    current_r = models.FloatField(null=True, blank=True)
    current_y = models.FloatField(null=True, blank=True)
    current_b = models.FloatField(null=True, blank=True)
    voltage_actual = models.FloatField(null=True, blank=True)
    speed_rpm = models.FloatField(null=True, blank=True)
    winding_temperature = models.FloatField(null=True, blank=True)  # °C
    bearing_temperature = models.FloatField(null=True, blank=True)  # °C
    vibration = models.FloatField(null=True, blank=True)  # mm/s
    power_consumption_kw = models.FloatField(default=0)
    power_factor = models.FloatField(null=True, blank=True)

    class Meta:
        verbose_name = "Motor"
        verbose_name_plural = "Motors"


# ============================================================
# Metering & Monitoring
# ============================================================


class EnergyMeter(BaseEquipment):
    """Energy Meters."""

    class MeterType(models.TextChoices):
        MAIN_INCOMER = "main_incomer", "Main Incomer"
        SUB_METER = "sub_meter", "Sub Meter"
        FEEDER = "feeder", "Feeder Meter"
        DG_METER = "dg_meter", "DG Meter"
        SOLAR_METER = "solar_meter", "Solar Meter"
        CHECK_METER = "check_meter", "Check Meter"

    meter_type = models.CharField(
        max_length=20, choices=MeterType.choices, default=MeterType.SUB_METER
    )
    ct_ratio = models.CharField(max_length=20, blank=True)  # e.g., "500/5"
    pt_ratio = models.CharField(max_length=20, blank=True)  # e.g., "11000/110"
    accuracy_class = models.CharField(max_length=10, blank=True)  # 0.2S, 0.5S, 1.0

    # Real-time readings
    voltage_r = models.FloatField(null=True, blank=True)
    voltage_y = models.FloatField(null=True, blank=True)
    voltage_b = models.FloatField(null=True, blank=True)
    voltage_ry = models.FloatField(null=True, blank=True)
    voltage_yb = models.FloatField(null=True, blank=True)
    voltage_br = models.FloatField(null=True, blank=True)
    current_r = models.FloatField(null=True, blank=True)
    current_y = models.FloatField(null=True, blank=True)
    current_b = models.FloatField(null=True, blank=True)
    power_kw = models.FloatField(default=0)
    power_kva = models.FloatField(default=0)
    power_kvar = models.FloatField(default=0)
    power_factor = models.FloatField(null=True, blank=True)
    frequency = models.FloatField(null=True, blank=True)
    total_kwh = models.FloatField(default=0)
    total_kvah = models.FloatField(default=0)
    max_demand_kw = models.FloatField(default=0)
    max_demand_kva = models.FloatField(default=0)

    class Meta:
        verbose_name = "Energy Meter"
        verbose_name_plural = "Energy Meters"


# ============================================================
# Alerts & Events
# ============================================================


class Alert(models.Model):
    """Equipment alerts and alarms."""

    class Severity(models.TextChoices):
        CRITICAL = "critical", "Critical"
        HIGH = "high", "High"
        MEDIUM = "medium", "Medium"
        LOW = "low", "Low"
        INFO = "info", "Info"

    class AlertType(models.TextChoices):
        THRESHOLD = "threshold", "Threshold Breach"
        FAULT = "fault", "Equipment Fault"
        MAINTENANCE = "maintenance", "Maintenance Due"
        COMMUNICATION = "communication", "Communication Loss"
        SAFETY = "safety", "Safety Alert"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    equipment_type = models.CharField(max_length=50)
    equipment_id = models.CharField(max_length=50, db_index=True)
    equipment_name = models.CharField(max_length=200)

    severity = models.CharField(max_length=20, choices=Severity.choices)
    alert_type = models.CharField(max_length=20, choices=AlertType.choices)
    message = models.TextField()
    parameter = models.CharField(max_length=100, blank=True)
    value = models.FloatField(null=True, blank=True)
    threshold = models.FloatField(null=True, blank=True)
    unit = models.CharField(max_length=20, blank=True)

    triggered_at = models.DateTimeField(default=timezone.now)
    acknowledged = models.BooleanField(default=False)
    acknowledged_by = models.CharField(max_length=100, blank=True)
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    resolved = models.BooleanField(default=False)
    resolved_at = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-triggered_at"]
        verbose_name = "Alert"
        verbose_name_plural = "Alerts"

    def __str__(self):
        return f"[{self.severity}] {self.equipment_name}: {self.message[:50]}"


# ============================================================
# Maintenance Records
# ============================================================


class MaintenanceRecord(models.Model):
    """Equipment maintenance history."""

    class MaintenanceType(models.TextChoices):
        PREVENTIVE = "preventive", "Preventive Maintenance"
        CORRECTIVE = "corrective", "Corrective Maintenance"
        BREAKDOWN = "breakdown", "Breakdown Maintenance"
        PREDICTIVE = "predictive", "Predictive Maintenance"
        INSPECTION = "inspection", "Inspection"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    equipment_type = models.CharField(max_length=50)
    equipment_id = models.CharField(max_length=50, db_index=True)
    equipment_name = models.CharField(max_length=200)

    maintenance_type = models.CharField(max_length=20, choices=MaintenanceType.choices)
    description = models.TextField()
    work_done = models.TextField(blank=True)
    parts_replaced = models.TextField(blank=True)
    cost = models.FloatField(null=True, blank=True)
    technician = models.CharField(max_length=100, blank=True)
    vendor = models.CharField(max_length=200, blank=True)

    scheduled_date = models.DateTimeField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    downtime_hours = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Maintenance Record"
        verbose_name_plural = "Maintenance Records"

    def __str__(self):
        return f"{self.equipment_name} - {self.maintenance_type} ({self.created_at.date()})"
