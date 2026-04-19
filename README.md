[![Crates.io](https://img.shields.io/crates/v/sigmoid-q15.svg)](https://crates.io/crates/sigmoid-q15)
[![Docs.rs](https://docs.rs/sigmoid-q15/badge.svg)](https://docs.rs/sigmoid-q15)
[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

# sigmoid-q15

Fonction d'activation **Sigmoid** (`1 / (1 + e^-x)`) en virgule fixe **Q15** pour systèmes embarqués `no_std`, **testé sur pico 2040 zero**.

Utilise [`embedded-exp`](https://crates.io/crates/embedded-exp) pour le calcul de `e^x`.

## Caractéristiques

- `#![no_std]` : aucune dépendance à la bibliothèque standard.
- `#![forbid(unsafe_code)]` : safety garantie par Rust.
- Arithmétique entière pure : pas de flottants, pas de `libm`.
- Compatible **RP2040** (Cortex-M0+) et **RP2350** (Cortex-M33).
- Temps d'exécution **constant** (déterministe) : idéal pour les noyaux temps réel.
- Zéro allocation dynamique : traitement in-place.

## Format Q15

En Q15, un `i16` représente un réel dans `[-1.0, 1.0[` :

```
valeur_réelle = valeur_i16 / 32768.0
```

| `i16`   | Valeur réelle |
|---------|--------------|
| `-32768`| -1.0         |
| `0`     | 0.0          |
| `16384` | 0.5          |
| `32767` | ≈ 1.0        |

## Algorithme

La sigmoid exploite une propriété de symétrie pour toujours rester dans le domaine valide de `embedded-exp` (valeurs négatives uniquement) :

```
sigmoid(x)  = 1 / (1 + e^(-x))   si x > 0  →  -x < 0 ✓
sigmoid(0)  = 0.5                  exact
sigmoid(x)  = 1 - sigmoid(-x)     si x < 0  →  symétrie ✓
```

## Utilisation

```toml
[dependencies]
sigmoid-q15 = "0.1.0"
```

```rust
use sigmoid_q15::{sigmoid_q15, sigmoid_slice_q15};

// sigmoid(0) = 0.5 → 16384 en Q15
assert_eq!(sigmoid_q15(0), 16384);

// Valeurs négatives → sortie < 0.5
assert!(sigmoid_q15(-16384) < 16384);

// Valeurs positives → sortie > 0.5
assert!(sigmoid_q15(16384) > 16384);

// In-place sur un slice (économise la RAM du MCU)
let mut buf = [-32768i16, 0, 32767];
sigmoid_slice_q15(&mut buf);
assert!(buf[0] > 0 && buf[0] < 16384);
assert_eq!(buf[1], 16384);
assert!(buf[2] > 16384);
```

## Cas limites garantis

| Entrée      |Sortie Q15 | Valeur réelle  | Raison                        |
|-------------|-----------|----------------|-------------------------------|
| `i16::MIN`  | ≈ 8808    | ≈ 0.2689       | sigmoid(-1.0)                 |
| `-1`        | < 16384   | < 0.5          | négatif → en dessous de 0.5   |
| `0`         | 16384     | = 0.5 exact    | sigmoid(0) = 0.5              |
| `1`         | > 16384   | > 0.5          | positif → au dessus de 0.5    |
| `i16::MAX`  | ≈ 23960   | ≈ 0.7311       | sigmoid(≈1.0)                 |

La sortie est **toujours** dans `[1, 32767]`  jamais zéro, jamais négatif.

## Ecosystème

Cette crate fait partie d'un écosystème d'activation Q15 pour MCU :

| Crate | Fonction |
|-------|----------|
| [`embedded-exp`](https://crates.io/crates/embedded-exp) | `e^x` en Q15 (dépendance) |
| [`relu-q15`](https://crates.io/crates/relu-q15) | ReLU en Q15 |
| [`sigmoid-q15`](https://crates.io/crates/sigmoid-q15) | Sigmoid en Q15 |


# Exemple Sur pico 2040 zero Embassy  
Affichage avec la Oled : Utilise [`embassy-ssd1306`](https://crates.io/crates/embassy-ssd1306) 


````rust
#![no_std]
#![no_main]

use cortex_m_rt as _;
use embassy_executor::Spawner;
use embassy_rp::i2c::{Config as I2cConfig, I2c, Async};
use embassy_time::Timer; 
use {panic_halt as _, embassy_rp as _};

use embassy_ssd1306::Ssd1306;
use embassy_sync::blocking_mutex::raw::NoopRawMutex;
use embassy_sync::mutex::Mutex;
use embassy_embedded_hal::shared_bus::asynch::i2c::I2cDevice;

use embassy_rp::bind_interrupts;
use embassy_rp::peripherals::I2C0; 
use rp2040_linker as _; 

// IMPORT DE LA NOUVELLE CRATE
use sigmoid_q15::sigmoid_q15;

bind_interrupts!(struct Irqs {
    I2C0_IRQ => embassy_rp::i2c::InterruptHandler<I2C0>;
});

#[embassy_executor::task]
async fn system_task(
    mut oled: Ssd1306<I2cDevice<'static, NoopRawMutex, I2c<'static, I2C0, Async>>>
) {
    if let Ok(_) = oled.init().await {
        oled.clear();
        oled.draw_rect(0, 0, 127, 63, true);
        let _ = oled.flush().await;
    }

    // Valeurs clés pour tester la courbe Sigmoïde
    // -32768 (-1.0), -16384 (-0.5), 0 (0.0), 16384 (0.5), 32767 (~1.0)
    let test_points: [i16; 5] = [i16::MIN, -16384, 0, 16384, i16::MAX];
    let mut idx = 0;

    loop {
        oled.clear();
        oled.draw_rect(0, 0, 127, 63, true);

        let x_input = test_points[idx];
        let y_output = sigmoid_q15(x_input);

        //  AFFICHAGE 
        oled.draw_str(10, 1, b"Sigmoid Q15");
        
        // Entrée (In)
        oled.draw_str(10, 3, b"In :");
        oled.draw_i16(50, 3, x_input);

        // Sortie (Out)
        oled.draw_str(10, 5, b"Out:");
        oled.draw_i16(50, 5, y_output);

        // Indicateur visuel du milieu (0.5 = 16384)
        if y_output == 16384 {
            oled.draw_str(90, 5, b"[MID]");
        }

        let _ = oled.flush().await;
        
        idx = (idx + 1) % test_points.len();
        Timer::after_secs(2).await;
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(embassy_rp::config::Config::default());
    
    let mut i2c_config = I2cConfig::default();
    i2c_config.frequency = 400_000; 
    
    let i2c_bus = I2c::new_async(p.I2C0, p.PIN_9, p.PIN_8, Irqs, i2c_config);

    static I2C_BUS: static_cell::StaticCell<Mutex<NoopRawMutex, I2c<'static, I2C0, Async>>> = static_cell::StaticCell::new();
    let i2c_mutex = I2C_BUS.init(Mutex::new(i2c_bus));

    let i2c_dev_oled = I2cDevice::new(i2c_mutex);
    let oled = Ssd1306::new(i2c_dev_oled, 0x3C);

    spawner.spawn(system_task(oled)).unwrap();
}
````

## Licence

GPL-2.0-or-later : voir [LICENSE](LICENSE).

# 🦅 À propos

Développé et testé par Jorge Andre Castro