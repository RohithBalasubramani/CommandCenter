from django.contrib import admin
from .models import (
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


@admin.register(Transformer)
class TransformerAdmin(admin.ModelAdmin):
    list_display = ["equipment_id", "name", "capacity_kva", "status", "health_score", "load_percent"]
    list_filter = ["status", "criticality", "transformer_type"]
    search_fields = ["equipment_id", "name", "location"]


@admin.register(DieselGenerator)
class DieselGeneratorAdmin(admin.ModelAdmin):
    list_display = ["equipment_id", "name", "capacity_kva", "status", "fuel_level_percent"]
    list_filter = ["status", "criticality"]
    search_fields = ["equipment_id", "name"]


@admin.register(ElectricalPanel)
class ElectricalPanelAdmin(admin.ModelAdmin):
    list_display = ["equipment_id", "name", "panel_type", "current_rating", "status"]
    list_filter = ["status", "panel_type", "criticality"]
    search_fields = ["equipment_id", "name"]


@admin.register(UPS)
class UPSAdmin(admin.ModelAdmin):
    list_display = ["equipment_id", "name", "capacity_kva", "status", "battery_percent", "load_percent"]
    list_filter = ["status", "ups_type", "criticality"]
    search_fields = ["equipment_id", "name"]


@admin.register(Chiller)
class ChillerAdmin(admin.ModelAdmin):
    list_display = ["equipment_id", "name", "capacity_tr", "chiller_type", "status", "load_percent"]
    list_filter = ["status", "chiller_type", "criticality"]
    search_fields = ["equipment_id", "name"]


@admin.register(AHU)
class AHUAdmin(admin.ModelAdmin):
    list_display = ["equipment_id", "name", "capacity_cfm", "status", "supply_air_temp"]
    list_filter = ["status", "ahu_type", "criticality", "building"]
    search_fields = ["equipment_id", "name", "location"]


@admin.register(CoolingTower)
class CoolingTowerAdmin(admin.ModelAdmin):
    list_display = ["equipment_id", "name", "capacity_tr", "status", "inlet_water_temp", "outlet_water_temp"]
    list_filter = ["status", "tower_type", "criticality"]
    search_fields = ["equipment_id", "name"]


@admin.register(Pump)
class PumpAdmin(admin.ModelAdmin):
    list_display = ["equipment_id", "name", "pump_type", "flow_rate", "status", "motor_kw"]
    list_filter = ["status", "pump_type", "criticality"]
    search_fields = ["equipment_id", "name"]


@admin.register(Compressor)
class CompressorAdmin(admin.ModelAdmin):
    list_display = ["equipment_id", "name", "capacity_cfm", "status", "load_percent"]
    list_filter = ["status", "compressor_type", "criticality"]
    search_fields = ["equipment_id", "name"]


@admin.register(Motor)
class MotorAdmin(admin.ModelAdmin):
    list_display = ["equipment_id", "name", "power_kw", "status", "rpm"]
    list_filter = ["status", "motor_type", "criticality", "efficiency_class"]
    search_fields = ["equipment_id", "name"]


@admin.register(EnergyMeter)
class EnergyMeterAdmin(admin.ModelAdmin):
    list_display = ["equipment_id", "name", "meter_type", "power_kw", "total_kwh"]
    list_filter = ["status", "meter_type", "criticality"]
    search_fields = ["equipment_id", "name", "location"]


@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ["equipment_name", "severity", "alert_type", "message", "triggered_at", "acknowledged", "resolved"]
    list_filter = ["severity", "alert_type", "acknowledged", "resolved"]
    search_fields = ["equipment_id", "equipment_name", "message"]


@admin.register(MaintenanceRecord)
class MaintenanceRecordAdmin(admin.ModelAdmin):
    list_display = ["equipment_name", "maintenance_type", "description", "completed_at", "technician"]
    list_filter = ["maintenance_type", "equipment_type"]
    search_fields = ["equipment_id", "equipment_name", "description"]
