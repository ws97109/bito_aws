import type { SubgraphResponse } from '../types/index';

// Helper to build a subgraph centered on a fraud node
function buildSubgraph(centerId: number, neighbors: number[], fraudNeighbors: Set<number>): SubgraphResponse {
  const nodes = [
    { user_id: centerId, risk_score: 0.95, status: 1 as const },
    ...neighbors.map((uid, i) => ({
      user_id: uid,
      risk_score: fraudNeighbors.has(uid) ? 0.6 + (i % 4) * 0.08 : 0.1 + (i % 5) * 0.06,
      status: (fraudNeighbors.has(uid) ? 1 : 0) as 0 | 1,
    })),
  ];

  const relations: Array<'R1' | 'R2' | 'R3'> = ['R1', 'R2', 'R3'];
  const edges = neighbors.map((uid, i) => ({
    source: centerId,
    target: uid,
    relation_type: relations[i % 3],
  }));

  // Add some neighbor-to-neighbor edges
  for (let i = 0; i < neighbors.length - 1; i += 3) {
    edges.push({
      source: neighbors[i],
      target: neighbors[i + 1],
      relation_type: relations[(i + 1) % 3],
    });
  }

  return { nodes, edges };
}

export const mockSubgraphs: Map<number, SubgraphResponse> = new Map([
  [1042, buildSubgraph(1042,
    [1187, 1356, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013],
    new Set([1187, 1356]))],

  [1187, buildSubgraph(1187,
    [1042, 1093, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027],
    new Set([1042, 1093]))],

  [1356, buildSubgraph(1356,
    [1274, 1511, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042],
    new Set([1274, 1511]))],

  [1093, buildSubgraph(1093,
    [1128, 1463, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055],
    new Set([1128, 1463]))],

  [1274, buildSubgraph(1274,
    [1319, 1076, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069],
    new Set([1319, 1076]))],

  [1511, buildSubgraph(1511,
    [1582, 1234, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085],
    new Set([1582, 1234]))],

  [1128, buildSubgraph(1128,
    [1407, 1155, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098],
    new Set([1407, 1155]))],

  [1463, buildSubgraph(1463,
    [1638, 1291, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113],
    new Set([1638, 1291]))],

  [1319, buildSubgraph(1319,
    [1174, 1523, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2125, 2126, 2127, 2128, 2129],
    new Set([1174, 1523]))],

  [1076, buildSubgraph(1076,
    [1389, 1061, 2130, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142, 2143],
    new Set([1389, 1061]))],

  [1582, buildSubgraph(1582,
    [1445, 1217, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2152, 2153, 2154, 2155, 2156, 2157, 2158, 2159, 2160],
    new Set([1445, 1217]))],

  [1234, buildSubgraph(1234,
    [1702, 1336, 2161, 2162, 2163, 2164, 2165, 2166, 2167, 2168, 2169, 2170, 2171, 2172, 2173, 2174],
    new Set([1702, 1336]))],
]);

const defaultSubgraph: SubgraphResponse = buildSubgraph(9999,
  [9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008, 9009, 9010, 9011, 9012, 9013, 9014, 9015],
  new Set([9001, 9002]));

export function getSubgraphData(userId: number, _hops: number): SubgraphResponse {
  return mockSubgraphs.get(userId) ?? defaultSubgraph;
}
