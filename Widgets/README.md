# Scenario Reference Database

Portable SQLite export of all scenario-related tables from the Design Workbench.

## File

- **`scenarios.sqlite3`** - standalone database (~1.4 MB)

## Tables

| Table | Rows | Description |
|---|---|---|
| `workbench_scenarioproduct` | 2 | Product verticals (e.g. ERP, MES) |
| `workbench_scenario` | 23 | Core scenario definitions (slug, name, tags, density targets, validation) |
| `workbench_scenariofixture` | 84 | Content fixtures each scenario can render |
| `workbench_scenariocomponentreference` | 23 | Component-to-scenario mappings |
| `workbench_scenariocontrolgroup` | 23 | Logical groupings of interactive controls |
| `workbench_scenariocontrol` | 36 | Individual controls (boolean, select, number, text, range) |
| `workbench_scenariobranch` | 2 | Branch contexts for snapshot comparisons |
| `workbench_scenariosnapshot` | 4 | Snapshot metadata per branch |
| `workbench_scenarioextract` | 23 | Raw component scenario extraction payloads |
| `workbench_scenarioaianalysis` | 0 | Cached AI validation recommendations |

## Relationships

```
ScenarioProduct
  └── Scenario
        ├── ScenarioFixture
        ├── ScenarioComponentReference
        ├── ScenarioControlGroup
        │     └── ScenarioControl
        ├── ScenarioExtract
        └── ScenarioAiAnalysis

ScenarioBranch
  └── ScenarioSnapshot
```

## Quick Access

```bash
# Open with sqlite3 CLI
sqlite3 scenarios.sqlite3

# List all scenarios
SELECT slug, name, summary FROM workbench_scenario;

# Scenarios with their product
SELECT s.name, p.label
FROM workbench_scenario s
JOIN workbench_scenarioproduct p ON s.product_id = p.id;

# Fixtures per scenario
SELECT s.slug, COUNT(f.id) AS fixtures
FROM workbench_scenario s
LEFT JOIN workbench_scenariofixture f ON f.scenario_id = s.id
GROUP BY s.slug;
```

## Source

Exported from `backend/db.sqlite3` on 2026-01-30.
