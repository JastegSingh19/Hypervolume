// src/hypervolume_random.rs

// Cargo.toml dependencies needed:
// rayon = "1.8"
// plotters = "0.3"

use rayon::prelude::*;
use std::time::Instant;
use plotters::prelude::*;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

type Point = Vec<f64>;
type PointSet = Vec<Point>;

#[derive(Clone)]
pub struct BenchmarkResult {
    pub dimension: usize,
    pub num_points: usize,
    pub hv: f64,
    // Means
    pub serial_time: f64,
    pub parallel_lvl1: f64,
    pub parallel_lvl2: f64,
    // Standard Deviations
    pub serial_std: f64,
    pub parallel_lvl1_std: f64,
    pub parallel_lvl2_std: f64,
}

// ============================================================================
// RANDOM NUMBER GENERATOR (LCG)
// ============================================================================
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 33) as f64 / 2147483648.0
    }

    fn range(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.next_f64()
    }
}

// ============================================================================
// EXPOSED WRAPPER
// ============================================================================

pub fn run_benchmark_selected() -> Vec<BenchmarkResult> {
    run_benchmark_grid()
}

// ============================================================================
// HYPERVOLUME CORE
// ============================================================================

pub fn calculate_hv_serial(points: &PointSet, ref_point: &Point) -> f64 {
    wfg_recursive(points, ref_point)
}

pub fn calculate_hv_parallel_lvl1(points: &PointSet, ref_point: &Point) -> f64 {
    let n = points.len();
    (0..n)
        .into_par_iter()
        .map(|i| exclusive_hv(&points[i], &points[i + 1..], ref_point))
        .sum()
}

pub fn calculate_hv_parallel_lvl2(points: &PointSet, ref_point: &Point) -> f64 {
    let n = points.len();
    (0..n)
        .into_par_iter()
        .map(|i| exclusive_hv_parallel(&points[i], &points[i + 1..], ref_point))
        .sum()
}

// ============================================================================
// RECURSIVE COMPUTATION HELPERS
// ============================================================================

fn wfg_recursive(points: &PointSet, ref_point: &Point) -> f64 {
    let n = points.len();
    let dim = ref_point.len();

    if n == 0 { return 0.0; }
    if dim == 1 { return (points[0][0] - ref_point[0]).max(0.0); }
    if dim == 2 { return hypervolume_2d(points, ref_point); }

    (0..n)
        .map(|i| exclusive_hv(&points[i], &points[i + 1..], ref_point))
        .sum()
}

fn exclusive_hv(point: &Point, rest: &[Point], ref_point: &Point) -> f64 {
    let inclusive: f64 = point.iter().zip(ref_point.iter()).map(|(p, r)| p - r).product();
    if rest.is_empty() { return inclusive; }

    let limited = limit_set(rest, point);
    let nd = get_non_dominated(&limited);
    if nd.is_empty() { return inclusive; }

    inclusive - wfg_recursive(&nd, ref_point)
}

fn exclusive_hv_parallel(point: &Point, rest: &[Point], ref_point: &Point) -> f64 {
    let inclusive: f64 = point.iter().zip(ref_point.iter()).map(|(p, r)| p - r).product();
    if rest.is_empty() { return inclusive; }

    let limited = limit_set(rest, point);
    let nd = get_non_dominated(&limited);
    if nd.is_empty() { return inclusive; }

    let sub: f64 = (0..nd.len())
        .into_par_iter()
        .map(|i| exclusive_hv(&nd[i], &nd[i + 1..], ref_point))
        .sum();

    inclusive - sub
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn limit_set(points: &[Point], limit: &Point) -> PointSet {
    points.iter().map(|p| p.iter().zip(limit.iter()).map(|(a, b)| a.min(*b)).collect()).collect()
}

fn get_non_dominated(points: &PointSet) -> PointSet {
    let n = points.len();
    let mut dom = vec![false; n];
    for i in 0..n {
        if dom[i] { continue; }
        for j in 0..n {
            if i == j || dom[j] { continue; }
            if dominates(&points[j], &points[i]) { dom[i] = true; break; }
            else if dominates(&points[i], &points[j]) { dom[j] = true; }
        }
    }
    points.iter().enumerate().filter(|(i, _)| !dom[*i]).map(|(_, p)| p.clone()).collect()
}

fn dominates(p: &Point, q: &Point) -> bool {
    let ge = p.iter().zip(q.iter()).all(|(pi, qi)| pi >= qi);
    let gt = p.iter().zip(q.iter()).any(|(pi, qi)| pi > qi);
    ge && gt
}

fn generate_structured_dataset(n_points: usize, dim: usize) -> PointSet {
    let mut rng = SimpleRng::new(12345 + n_points as u64);
    let mut points = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let mut p = Vec::with_capacity(dim);
        p.push((i + 1) as f64);              // Coord 1: 1..N
        p.push((n_points - i) as f64);       // Coord 2: N..1
        for _ in 2..dim {
            p.push(rng.range(1.0, n_points as f64)); // Others: Random
        }
        points.push(p);
    }
    points
}

fn hypervolume_2d(points: &PointSet, reference_point: &Point) -> f64 {
    let mut sorted = points.clone();
    sorted.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
    let mut hv = 0.0;
    let mut prev_x = reference_point[0];
    for p in sorted {
        hv += (p[0] - prev_x) * (p[1] - reference_point[1]);
        prev_x = p[0];
    }
    hv
}

fn mean_and_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    if n <= 1.0 { return (v[0], 0.0); }
    let mean = v.iter().sum::<f64>() / n;
    let variance = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, variance.sqrt())
}

// ============================================================================
// BENCHMARK + TABLE PRINT + PLOT
// ============================================================================

pub fn run_benchmark_grid() -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    
    let dimensions = vec![2,4,8,9];
    let ns = (100..=1000).step_by(100).collect::<Vec<usize>>();
    let runs = 5; // UPDATED: 5 runs

    println!("\n========================================================================================================");
    println!("DIM |   N  |   Hypervolume  |      Serial(ms)      |        P1(ms)        |        P2(ms)");
    println!("--------------------------------------------------------------------------------------------------------");

    for &dim in &dimensions {
        for &n in &ns {
            let pts = generate_structured_dataset(n, dim);
            let reference = vec![0.0; dim]; 

            let hv_serial = calculate_hv_serial(&pts, &reference);
            
            // Validity check (only on first iteration of N to save time)
            if n == 100 {
                let hv_p1 = calculate_hv_parallel_lvl1(&pts, &reference);
                if (hv_serial - hv_p1).abs() > hv_serial.abs() * 1e-9 {
                    println!("❌ Error: HV mismatch at Dim={} N={}", dim, n);
                }
            }

            let mut t_serial = vec![];
            let mut t_p1 = vec![];
            let mut t_p2 = vec![];

            for _ in 0..runs {
                let t = Instant::now();
                calculate_hv_serial(&pts, &reference);
                t_serial.push(t.elapsed().as_secs_f64() * 1000.0);

                let t = Instant::now();
                calculate_hv_parallel_lvl1(&pts, &reference);
                t_p1.push(t.elapsed().as_secs_f64() * 1000.0);

                let t = Instant::now();
                calculate_hv_parallel_lvl2(&pts, &reference);
                t_p2.push(t.elapsed().as_secs_f64() * 1000.0);
            }

            let (s_mean, s_std) = mean_and_std(&t_serial);
            let (p1_mean, p1_std) = mean_and_std(&t_p1);
            let (p2_mean, p2_std) = mean_and_std(&t_p2);

            let r = BenchmarkResult {
                dimension: dim,
                num_points: n,
                hv: hv_serial,
                serial_time: s_mean,
                serial_std: s_std,
                parallel_lvl1: p1_mean,
                parallel_lvl1_std: p1_std,
                parallel_lvl2: p2_mean,
                parallel_lvl2_std: p2_std,
            };

            // Format: Mean ± Std
            println!(
                "{:3} | {:4} | {:14.2e} | {:8.2} ± {:<6.2} | {:8.2} ± {:<6.2} | {:8.2} ± {:<6.2}",
                r.dimension, r.num_points, r.hv,
                s_mean, s_std,
                p1_mean, p1_std,
                p2_mean, p2_std
            );

            results.push(r);
        }
    }

    results
}

pub fn plot(results: &[BenchmarkResult]) {
    let target_dim = 7;
    let filtered: Vec<&BenchmarkResult> = results.iter().filter(|r| r.dimension == target_dim).collect();

    if filtered.is_empty() {
        println!("No results for dimension {} to plot.", target_dim);
        return;
    }

    let root = BitMapBackend::new("benchmark_plot.png", (1200, 700)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let max_time = filtered
        .iter()
        .map(|r| r.serial_time.max(r.parallel_lvl1.max(r.parallel_lvl2)))
        .fold(0.0, f64::max);
    
    let max_n = filtered.last().unwrap().num_points;

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("Benchmark (Dim={}): Serial vs Parallel", target_dim), ("sans-serif", 30))
        .build_cartesian_2d(100..max_n, 0.0..max_time * 1.1)
        .unwrap();

    chart.configure_mesh()
        .x_desc("Number of Points (N)")
        .y_desc("Mean Time (ms)")
        .draw().unwrap();

    chart.draw_series(LineSeries::new(
        filtered.iter().map(|r| (r.num_points, r.serial_time)),
        &RED,
    )).unwrap()
        .label("Serial")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x+20, y)], RED));

    chart.draw_series(LineSeries::new(
        filtered.iter().map(|r| (r.num_points, r.parallel_lvl1)),
        &BLUE,
    )).unwrap()
        .label("Parallel Lvl 1")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x+20, y)], BLUE));

    chart.draw_series(LineSeries::new(
        filtered.iter().map(|r| (r.num_points, r.parallel_lvl2)),
        &GREEN,
    )).unwrap()
        .label("Parallel Lvl 2")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x+20, y)], GREEN));

    chart.configure_series_labels().border_style(&BLACK).draw().unwrap();
}