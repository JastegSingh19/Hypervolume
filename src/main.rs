// src/main.rs

// Declare the three files as modules
mod hypervolume_random;
mod hypervolume_random2;
mod hypervolume_worst_case;

fn main() {
    // --- RUN BENCHMARK 1 (Random) ---
    println!("\n🚀 Starting Hypervolume Random (Set 1)...");
    let results1 = hypervolume_random::run_benchmark_selected();
    if !results1.is_empty() {
        // Use the specific module's plot function
        hypervolume_random::plot(&results1);
        println!("✓ Random 1 Done. Graph saved (check hypervolume_random for filename).");
    }

    println!("\n----------------------------------------------------\n");

    // --- RUN BENCHMARK 2 (Random 2 - Higher Dimensions) ---
    println!("🚀 Starting Hypervolume Random (Set 2 - High Dim)...");
    let results2 = hypervolume_random2::run_benchmark_selected();
    if !results2.is_empty() {
        hypervolume_random2::plot(&results2);
        println!("✓ Random 2 Done. Graph saved.");
    }

    println!("\n----------------------------------------------------\n");

    // --- RUN BENCHMARK 3 (Worst Case) ---
    println!("🚀 Starting Hypervolume Worst Case (Cyclic Points)...");
    // Note: run_benchmark(max_dim, iterations)
    let results3 = hypervolume_worst_case::run_benchmark(15, 5); 
    if !results3.is_empty() {
        hypervolume_worst_case::plot(&results3);
        println!("✓ Worst Case Done. Graph saved.");
    }

    println!("\nAll benchmarks completed! 🏁");
}