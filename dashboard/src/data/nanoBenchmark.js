// Real numbers from results/nano_benchmark_REAL_v2.json: the same Defender
// model this replay uses, scoring 815 real ROAD validation windows on the ZGX
// Nano (GB10, host spark-3f6), via defender/nano_runner.py. Inference only;
// window collection time excluded. Copied, not recomputed, so the dashboard
// cites the same evidence as the README -- update by hand if it is re-run.
export const NANO_BENCHMARK = {
  device: "ZGX Nano (GB10)",
  host: "spark-3f6",
  architecture: "aarch64",
  pythonVersion: "3.12.3",
  windows: 815,
  latencyMsMean: 1.3166897073873758,
  latencyMsP95: 1.6446626279503107,
  latencyMsMax: 1.7310369876213372,
  windowsPerSecond: 759.4803805250644,
  windowSeconds: 1, // each window is 1 s of driving, so windows arrive at 1/s
  source: "results/nano_benchmark_REAL_v2.json",
};
