use plotters::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use std::time::Instant;

type Point = Vec<f64>;
type PointSet = Vec<Point>;

#[derive(Clone)]
pub struct BenchmarkResult {
    pub dimension: usize,
    pub num_points: usize,
    pub hv: f64,
    pub serial_time: f64,
    pub parallel_time: f64,
    pub serial_std: f64,
    pub parallel_std: f64,
}

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

    fn gen_index(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }

        (self.next_f64() * upper as f64).floor() as usize % upper
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.gen_index(i + 1);
            values.swap(i, j);
        }
    }
}

pub fn run_benchmark_selected(parallel_depth: usize) -> Vec<BenchmarkResult> {
    run_benchmark_grid(parallel_depth)
}

pub fn calculate_hv_serial(points: &PointSet, ref_point: &Point) -> f64 {
    wfg_recursive(points, ref_point)
}

pub fn calculate_hv_parallel(points: &PointSet, ref_point: &Point, parallel_depth: usize) -> f64 {
    if parallel_depth == 0 {
        return calculate_hv_serial(points, ref_point);
    }

    let n = points.len();
    (0..n)
        .into_par_iter()
        .map(|i| {
            exclusive_hv_with_depth(&points[i], &points[i + 1..], ref_point, parallel_depth - 1)
        })
        .sum()
}

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

fn exclusive_hv_with_depth(
    point: &Point,
    rest: &[Point],
    ref_point: &Point,
    parallel_depth: usize,
) -> f64 {
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

    let sub = if parallel_depth == 0 {
        wfg_recursive(&nd, ref_point)
    } else {
        (0..nd.len())
            .into_par_iter()
            .map(|i| exclusive_hv_with_depth(&nd[i], &nd[i + 1..], ref_point, parallel_depth - 1))
            .sum()
    };

    inclusive - sub
}

fn limit_set(points: &[Point], limit: &Point) -> PointSet {
    points
        .iter()
        .map(|p| p.iter().zip(limit.iter()).map(|(a, b)| a.min(*b)).collect())
        .collect()
}

fn get_non_dominated(points: &PointSet) -> PointSet {
    let n = points.len();
    let mut dom = vec![false; n];

    for i in 0..n {
        if dom[i] {
            continue;
        }
        for j in 0..n {
            if i == j || dom[j] {
                continue;
            }
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

fn generate_structured_dataset(n_points: usize, dim: usize) -> PointSet {
    const TOTAL: u16 = 1000;

    let mut rng = SimpleRng::new(12345 + n_points as u64 + (dim as u64 * 10_000));
    let mut seen: HashSet<Vec<u16>> = HashSet::with_capacity(n_points);
    let mut points = Vec::with_capacity(n_points);

    while points.len() < n_points {
        let mut cuts = HashSet::with_capacity(dim.saturating_sub(1));
        while cuts.len() + 1 < dim {
            let cut = 1 + rng.gen_index((TOTAL - 1) as usize) as u16;
            cuts.insert(cut);
        }

        let mut sorted_cuts: Vec<u16> = cuts.into_iter().collect();
        sorted_cuts.sort_unstable();

        let mut previous = 0u16;
        let mut coords = Vec::with_capacity(dim);

        for cut in sorted_cuts {
            coords.push(cut - previous);
            previous = cut;
        }
        coords.push(TOTAL - previous);

        rng.shuffle(&mut coords);

        if seen.insert(coords.clone()) {
            let point = coords
                .into_iter()
                .map(|value| value as f64 / TOTAL as f64)
                .collect();
            points.push(point);
        }
    }

    rng.shuffle(&mut points);
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

fn print_run_data(label: &str, serial: &[f64], parallel: &[f64]) {
    let serial_str = serial
        .iter()
        .map(|v| format!("{v:.3}"))
        .collect::<Vec<_>>()
        .join(", ");
    let parallel_str = parallel
        .iter()
        .map(|v| format!("{v:.3}"))
        .collect::<Vec<_>>()
        .join(", ");

    println!("  {} runs: serial=[{}] ms | parallel=[{}] ms", label, serial_str, parallel_str);
}

fn mean_and_std(v: &[f64]) -> (f64, f64) {
    let n = v.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    if n == 1 {
        return (v[0], 0.0);
    }

    let mean = v.iter().sum::<f64>() / n as f64;
    let variance = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    (mean, variance.sqrt())
}

fn print_sample_points(points: &PointSet, dim: usize, n: usize) {
    let sample_size = points.len().min(5);
    println!("Sample points for m={} n={}:", dim, n);
    for point in points.iter().take(sample_size) {
        let formatted = point
            .iter()
            .map(|value| format!("{value:.3}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!("[{}]", formatted);
    }
}

pub fn run_benchmark_grid(parallel_depth: usize) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    let dimensions = vec![2, 3, 4, 5, 6, 7, 8];
    let ns = (100..=1000).step_by(100).collect::<Vec<usize>>();
    let runs = 5;

    println!(
        "\n===================================================================================================="
    );
    println!(
        "DIM |   N  |   Hypervolume  |      Serial(ms)      | Parallel d={} (ms)",
        parallel_depth
    );
    println!(
        "----------------------------------------------------------------------------------------------------"
    );

    for &dim in &dimensions {
        for &n in &ns {
            let pts = generate_structured_dataset(n, dim);
            let reference = vec![0.0; dim];

            print_sample_points(&pts, dim, n);

            let hv_serial = calculate_hv_serial(&pts, &reference);

            if n == 100 {
                let hv_parallel = calculate_hv_parallel(&pts, &reference, parallel_depth);
                if (hv_serial - hv_parallel).abs() > hv_serial.abs() * 1e-9 {
                    println!("Error: HV mismatch at Dim={} N={}", dim, n);
                }
            }

            let mut t_serial = vec![];
            let mut t_parallel = vec![];

            for _ in 0..runs {
                let t = Instant::now();
                calculate_hv_serial(&pts, &reference);
                t_serial.push(t.elapsed().as_secs_f64() * 1000.0);

                let t = Instant::now();
                calculate_hv_parallel(&pts, &reference, parallel_depth);
                t_parallel.push(t.elapsed().as_secs_f64() * 1000.0);
            }

            print_run_data(&format!("Dim={} N={}", dim, n), &t_serial, &t_parallel);

            let (s_mean, s_std) = mean_and_std(&t_serial);
            let (p_mean, p_std) = mean_and_std(&t_parallel);

            let r = BenchmarkResult {
                dimension: dim,
                num_points: n,
                hv: hv_serial,
                serial_time: s_mean,
                parallel_time: p_mean,
                serial_std: s_std,
                parallel_std: p_std,
            };

            println!(
                "{:3} | {:4} | {:14.2e} | {:8.2} ± {:<6.2} | {:8.2} ± {:<6.2}",
                r.dimension,
                r.num_points,
                r.hv,
                r.serial_time,
                r.serial_std,
                r.parallel_time,
                r.parallel_std
            );

            results.push(r);
        }
    }

    results
}

pub fn plot(results: &[BenchmarkResult], parallel_depth: usize) {
    let target_dim = 8;
    let filtered: Vec<&BenchmarkResult> = results
        .iter()
        .filter(|r| r.dimension == target_dim)
        .collect();

    if filtered.is_empty() {
        println!("No results for dimension {} to plot.", target_dim);
        return;
    }

    let root = BitMapBackend::new("benchmark_plot_random.png", (1200, 700)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let max_time = filtered
        .iter()
        .map(|r| r.serial_time.max(r.parallel_time))
        .fold(0.0, f64::max);

    let max_n = filtered.last().unwrap().num_points;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "Benchmark (Dim={}): Serial vs Parallel Depth {}",
                target_dim, parallel_depth
            ),
            ("sans-serif", 30),
        )
        .build_cartesian_2d(100..max_n, 0.0..max_time * 1.1)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Number of Points (N)")
        .y_desc("Mean Time (ms)")
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            filtered.iter().map(|r| (r.num_points, r.serial_time)),
            &RED,
        ))
        .unwrap()
        .label("Serial")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .draw_series(LineSeries::new(
            filtered.iter().map(|r| (r.num_points, r.parallel_time)),
            &BLUE,
        ))
        .unwrap()
        .label(format!("Parallel depth {}", parallel_depth))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .draw()
        .unwrap();
}
