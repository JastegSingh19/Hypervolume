// Cargo.toml:
// [dependencies]
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
struct BenchmarkResult {
    dimension: usize,
    hv: f64,
    serial_time: f64,
    parallel_lvl1: f64,
    parallel_lvl2: f64,
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

    if n == 0 {
        return 0.0;
    }
    if dim == 1 {
        return (points[0][0] - ref_point[0]).max(0.0);
    }
    if dim == 2 {
        return hypervolume_2d(points, ref_point);
    }

    (0..n)
        .map(|i| exclusive_hv(&points[i], &points[i + 1..], ref_point))
        .sum()
}

fn exclusive_hv(point: &Point, rest: &[Point], ref_point: &Point) -> f64 {
    let inclusive: f64 = point
        .iter()
        .zip(ref_point.iter())
        .map(|(p, r)| p - r)
        .product();

    if rest.is_empty() {
        return inclusive;
    }

    let limited = limit_set(rest, point);
    let nd = get_non_dominated(&limited);

    if nd.is_empty() {
        return inclusive;
    }

    inclusive - wfg_recursive(&nd, ref_point)
}

fn exclusive_hv_parallel(point: &Point, rest: &[Point], ref_point: &Point) -> f64 {
    let inclusive: f64 = point
        .iter()
        .zip(ref_point.iter())
        .map(|(p, r)| p - r)
        .product();

    if rest.is_empty() {
        return inclusive;
    }

    let limited = limit_set(rest, point);
    let nd = get_non_dominated(&limited);

    if nd.is_empty() {
        return inclusive;
    }

    // parallel recursion for second-level
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
    points
        .iter()
        .map(|p| {
            p.iter()
                .zip(limit.iter())
                .map(|(a, b)| a.min(*b))
                .collect()
        })
        .collect()
}

fn get_non_dominated(points: &PointSet) -> PointSet {
    let n = points.len();
    let mut dom = vec![false; n];

    for i in 0..n {
        if dom[i] { continue; }
        for j in 0..n {
            if i == j || dom[j] { continue; }
            if dominates(&points[j], &points[i]) {
                dom[i] = true;
                break;
            } else if dominates(&points[i], &points[j]) {
                dom[j] = true;
            }
        }
    }

    points
        .iter()
        .enumerate()
        .filter(|(i, _)| !dom[*i])
        .map(|(_, p)| p.clone())
        .collect()
}

fn dominates(p: &Point, q: &Point) -> bool {
    let ge = p.iter().zip(q.iter()).all(|(pi, qi)| pi >= qi);
    let gt = p.iter().zip(q.iter()).any(|(pi, qi)| pi > qi);
    ge && gt
}

fn generate_cyclic_points(dim: usize) -> PointSet {
    let base: Vec<f64> = (1..=dim).map(|x| x as f64).collect();
    (0..dim)
        .map(|s| (0..dim).map(|i| base[(i + s) % dim]).collect())
        .collect()
}

// ============================================================================
// BENCHMARK + TABLE PRINT + PLOT
// ============================================================================

fn run_benchmark(max: usize, runs: usize) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    println!("\n====================================================");
    println!("DIM | Hypervolume | Serial(ms) | P1(ms) | P2(ms)");
    println!("----------------------------------------------------");

    for dim in 2..=max {
        let pts = generate_cyclic_points(dim);
        let reference = vec![0.0; dim];

        let hv_serial = calculate_hv_serial(&pts, &reference);
        let hv_p1 = calculate_hv_parallel_lvl1(&pts, &reference);
        let hv_p2 = calculate_hv_parallel_lvl2(&pts, &reference);

        let tolerance = hv_serial.abs() * 1e-12;
        if (hv_serial - hv_p1).abs() > tolerance || (hv_serial - hv_p2).abs() > tolerance {
            println!("\n❌ Hypervolume mismatch detected at dimension {}!", dim);
            println!("Serial:         {:.12}", hv_serial);
            println!("Parallel Lvl1:  {:.12}", hv_p1);
            println!("Parallel Lvl2:  {:.12}", hv_p2);
            println!("\nAborting benchmark to avoid invalid results.\n");
            std::process::exit(1);
        }

        let hv = hv_serial;


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

        let r = BenchmarkResult {
            dimension: dim,
            hv,
            serial_time: avg(&t_serial),
            parallel_lvl1: avg(&t_p1),
            parallel_lvl2: avg(&t_p2),
        };

        println!(
            "{:3} | {:12.0} | {:10.4} | {:7.4} | {:7.4}",
            r.dimension, r.hv, r.serial_time, r.parallel_lvl1, r.parallel_lvl2
        );

        results.push(r);
    }

    results
}

fn avg(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn plot(results: &[BenchmarkResult]) {
    let root = BitMapBackend::new("benchmark_plot.png", (1200, 700)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let max_time = results
        .iter()
        .map(|r| r.serial_time.max(r.parallel_lvl1.max(r.parallel_lvl2)))
        .fold(0.0, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Benchmark: Serial vs Level-Parallelism", ("sans-serif", 30))
        .build_cartesian_2d(2..results.len()+1, 0.0..max_time * 1.1)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart.draw_series(LineSeries::new(
        results.iter().map(|r| (r.dimension, r.serial_time)),
        &RED,
    )).unwrap()
        .label("Serial")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x+20, y)], RED));

    chart.draw_series(LineSeries::new(
        results.iter().map(|r| (r.dimension, r.parallel_lvl1)),
        &BLUE,
    )).unwrap()
        .label("Parallel-Lvl-1")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x+20, y)], BLUE));

    chart.draw_series(LineSeries::new(
        results.iter().map(|r| (r.dimension, r.parallel_lvl2)),
        &GREEN,
    )).unwrap()
        .label("Parallel-Lvl-2")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x+20, y)], GREEN));

    chart.configure_series_labels().border_style(&BLACK).draw().unwrap();
}

// ============================================================================
// MAIN
// ============================================================================
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

fn main() {
    let results = run_benchmark(30, 5);
    plot(&results);
    println!("\n✓ Done. Graph saved as benchmark_plot.png\n");
}
