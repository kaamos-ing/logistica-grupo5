"""Piloto CVRP exacto: seis entregas, depósito común, N rutas no vacías.

Adaptación del cuaderno DIMENSIÓN2_FINAL, celdas 17 y 21.
Las distancias son geodésicas y los tiempos son supuestos ilustrativos.
"""
from functools import lru_cache
from itertools import permutations
import numpy as np
import pandas as pd


def haversine(lat1, lon1, lat2, lon2):
    a1, a2 = np.radians(lat1), np.radians(lat2)
    x = np.sin((a2-a1)/2)**2 + np.cos(a1)*np.cos(a2)*np.sin(np.radians(lon2-lon1)/2)**2
    return 2*3958.7613*np.arcsin(np.sqrt(np.clip(x, 0, 1)))


def feasible(weights, n, capacity, nonempty=False):
    loads, assignments = [0.0]*n, [[] for _ in range(n)]
    def rec(i):
        if i == len(weights):
            return not nonempty or all(assignments)
        seen = set()
        for j in sorted(range(n), key=lambda k: (loads[k], k)):
            # Empty and nonempty trucks with equal load are not interchangeable.
            state = (loads[j], bool(assignments[j]))
            if state in seen:
                continue
            seen.add(state)
            if loads[j]+weights[i] <= capacity+1e-9:
                loads[j] += weights[i]; assignments[j].append(i)
                if rec(i+1):
                    return True
                loads[j] -= weights[i]; assignments[j].pop()
        return False
    return [r[:] for r in assignments] if rec(0) else None


def find_pilot(trips):
    for (date, origin), group in trips.groupby(['dispatch_date', 'pickup_facility'], sort=True):
        if len(group) < 6:
            continue
        c = group.sort_values(['weight_lbs', 'trip_id']).head(6).copy()
        coords = list(zip(c.destino_latitude.round(4), c.destino_longitude.round(4)))
        depot = (round(float(c.origen_latitude.iloc[0]), 4), round(float(c.origen_longitude.iloc[0]), 4))
        if c.delivery_facility.nunique() != 6 or len(set(coords)) != 6 or depot in coords:
            continue
        if feasible(c.weight_lbs.astype(int).tolist(), 3, 45000) is not None:
            return c.sort_values('delivery_scheduled').reset_index(drop=True)
    raise ValueError('No hay piloto histórico de seis destinos factible para tres camiones.')


def solve_pilot(pilot, capacity=45000, speed=50.0, service=0.5):
    if capacity <= 0 or speed <= 0 or service < 0:
        raise ValueError('Capacidad y velocidad deben ser positivas; servicio no negativo.')
    if len(pilot) != 6:
        raise ValueError('Este optimizador está acotado al piloto de seis pedidos.')
    origin = pilot.pickup_facility.iloc[0]
    if pilot.pickup_facility.nunique() != 1:
        raise ValueError('Las entregas deben compartir un depósito.')
    lat = np.r_[pilot.origen_latitude.iloc[0], pilot.destino_latitude.to_numpy()]
    lon = np.r_[pilot.origen_longitude.iloc[0], pilot.destino_longitude.to_numpy()]
    matrix = haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    weights = pilot.weight_lbs.astype(int).tolist()
    def distance(visits):
        seq = [0]+[v+1 for v in visits]+[0]
        return float(sum(matrix[a, b] for a, b in zip(seq[:-1], seq[1:])))
    best_subset = {}
    for mask in range(1, 1 << len(weights)):
        subset = tuple(i for i in range(len(weights)) if mask >> i & 1)
        if sum(weights[i] for i in subset) <= capacity:
            order = min(permutations(subset), key=distance)
            best_subset[mask] = (distance(order), order)
    @lru_cache(None)
    def dp(mask, n):
        if n == 0:
            return (0.0, ()) if mask == 0 else (float('inf'), ())
        if mask.bit_count() < n:
            return float('inf'), ()
        best = (float('inf'), ())
        sub = mask
        while sub:
            if sub in best_subset:
                prev, routes = dp(mask ^ sub, n-1)
                d, order = best_subset[sub]
                if d+prev < best[0]:
                    best = (d+prev, routes+(order,))
            sub = (sub-1) & mask
        return best
    rows, details, paths = [], [], {}
    for n in (3, 5):
        cost, optimized = dp((1 << len(weights))-1, n)
        base = feasible(weights, n, capacity, nonempty=True)
        if not np.isfinite(cost) or base is None:
            rows.append({'Camiones': n, 'Estado': 'Sin solución con esta capacidad'})
            continue
        base_cost = sum(distance(r) for r in base)
        ref_time = max(distance(r)/speed+len(r)*service for r in base)
        opt_time = max(distance(r)/speed+len(r)*service for r in optimized)
        rows.append({'Camiones': n, 'Estado': 'Factible', 'Referencia (mi)': base_cost,
                     'Optimizada (mi)': cost, 'Reducción (%)': (base_cost-cost)/base_cost*100 if base_cost else 0,
                     'Tiempo máximo referencia (h)': ref_time, 'Tiempo máximo optimizado (h)': opt_time})
        for kind, route_set in [('Referencia', base), ('Optimizada', optimized)]:
            paths[n, kind] = route_set
            assert len(route_set) == n and all(route_set)
            assert sorted(i for r in route_set for i in r) == list(range(6))
            for j, r in enumerate(route_set, 1):
                load = sum(weights[i] for i in r)
                assert load <= capacity
                details.append({'Camiones': n, 'Escenario': kind, 'Vehículo': j,
                    'Secuencia': ' → '.join([origin]+[pilot.delivery_facility.iloc[i] for i in r]+[origin]),
                    'Carga (lb)': load, 'Ocupación (%)': load/capacity*100,
                    'Distancia (mi)': distance(r), 'Tiempo estimado (h)': distance(r)/speed+len(r)*service})
    return {'comparison': pd.DataFrame(rows), 'details': pd.DataFrame(details),
            'paths': paths, 'lat': lat, 'lon': lon, 'matrix': matrix}
