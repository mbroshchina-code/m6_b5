# Supervisor-граф

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	supervisor(supervisor)
	researcher(researcher)
	writer(writer)
	__end__([<p>__end__</p>]):::last
	__start__ --> supervisor;
	researcher -.-> supervisor;
	supervisor -.-> __end__;
	supervisor -.-> researcher;
	supervisor -.-> writer;
	writer -.-> supervisor;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```
## Схема для редакторов без поддержки расширенного оформления

Те же узлы и переходы из draw_mermaid(), без настроек оформления.

```mermaid
graph TD
    start_node(["Начало"])
    supervisor["Supervisor"]
    researcher["Researcher"]
    writer["Writer"]
    end_node(["Конец"])

    start_node --> supervisor
    researcher -.-> supervisor
    supervisor -.-> end_node
    supervisor -.-> researcher
    supervisor -.-> writer
    writer -.-> supervisor
```