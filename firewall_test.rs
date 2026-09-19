 use std::process::Command;

fn main() {
    // Synthetic test: Invoking host process commands
    let output = Command::new("whoami").output();
    println!("{:?}", output);
} 
