use std::process::Command;

fn main() {
    // Malicious agent attempting unauthorized process execution
    let _ = Command::new("whoami").output();
}
