"""Capa de ejecución para paper trading con Alpaca.

Fase 1 del roadmap: construcción de órdenes multi-leg y controles de riesgo
previos al envío. Nada en este paquete envía órdenes por sí mismo: el envío, el
estado y el loop intradía llegan en las fases siguientes, una vez resueltas las
pruebas de ``scripts/paper_probe.py``.
"""
