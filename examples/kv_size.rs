use kvflow_sim::model::profiles;
use kvflow_sim::transfer::format_bytes;

fn main() {
    let models = [
        profiles::llama_8b_bf16_gqa(),
        profiles::llama_70b_bf16_gqa(),
    ];
    let contexts = [1024, 4096, 8192, 32768, 131072];

    println!("model,context_tokens,kv_per_token,kv_total");
    for model in models {
        for tokens in contexts {
            println!(
                "{},{},{},{}",
                model.model_id,
                tokens,
                format_bytes(model.kv_bytes_per_token()),
                format_bytes(model.kv_bytes(tokens)),
            );
        }
    }
}
