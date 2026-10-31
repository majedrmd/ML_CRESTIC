# \# Anomaly detection on a Modbus TCP industrial cell (LSTM + secured pipeline)

# 

# Real-time anomaly detection for a simulated pick-and-place cell (Factory I/O) controlled over \*\*Modbus TCP\*\*. An LSTM learns the normal sequence of sensor and actuator states. At runtime it predicts the next state and flags any signal that differs from what the machine actually does. The data path is protected by an IP-allowlist proxy and encrypted file transfer.

# 

# \*Internship project at CReSTIC (Université de Reims Champagne-Ardenne), 2026.\*

# 

# !\[LSTM training curves](anomaly\_detection/figures/lstm\_training\_curves.png)

# 

# \## Results

# 

# | Model | Input | Predicts | Accuracy |

# |---|---|---|---|

# | \*\*LSTM\*\* | last 10 states × 9 signals | next state (9 signals) | \*\*98.94%\*\* |

# | MLP baseline | current 4 sensors | 5 actuators | \~81% |

# 

# \- \*\*Accuracy\*\* here is flat accuracy: the share of individual signals predicted correctly.

# \- The LSTM is evaluated on a chronological split (no shuffling, so it never sees the future). The MLP baseline uses a random 80/20 split, so the comparison is indicative, not strictly like-for-like.

# 

# \## How it works

# 

# ```mermaid

# flowchart LR

# &#x20;   T\[Tutor machine<br/>secure\_client\_xyz.py] -->|Modbus TCP :5020| P\[IP-allowlist proxy<br/>secure\_server\_xyz.py]

# &#x20;   P -->|allowed IPs only| F\[Factory I/O<br/>Modbus server :502]

# &#x20;   P -->|records states| C\[(CSV log)]

# &#x20;   C --> S\[csv\_sender.py<br/>Fernet encryption] -->|TCP :5025| O\[observer.py<br/>decrypts and stores]

# &#x20;   F -->|live states| D\[online\_diagnosis.py<br/>LSTM]

# ```

# 

# 1\. \*\*Acquisition.\*\* `acquisition/record\_states.py` polls the cell over Modbus and logs every state change to CSV.

# 2\. \*\*Training.\*\* `anomaly\_detection/train\_model.py` builds sliding windows of 10 states and trains the LSTM.

# 3\. \*\*Live detection.\*\* `anomaly\_detection/online\_diagnosis.py` reads the cell every 100 ms, predicts the next state, and reports which signals deviate.

# 4\. \*\*Security.\*\* The proxy only forwards Modbus traffic from allow-listed IPs and logs every attempt. Recorded CSVs are encrypted with Fernet before being sent to the observer machine.

# 

# \## Data

# 

# Factory I/O \*Pick and Place XYZ\* scene with \*\*9 binary signals\*\*:

# 

# \- \*\*Sensors (4):\*\* `box\_at\_place`, `part\_at\_place`, `detected`, `c\_limit`

# \- \*\*Actuators (5):\*\* `box\_conveyor`, `part\_conveyor`, `exit\_conveyor`, `c\_plus`, `grap`

# 

# The training file has 1,336 states, which gives 1,326 sequences of 10. Raw recordings are in `data/raw/`.

# 

# \## Model

# 

# \- 1-layer LSTM with 64 hidden units, sigmoid output, binary cross-entropy loss

# \- Adam (learning rate 1e-3), batch size 32, early stopping (patience 10)

# \- About 19.8k parameters, small enough to run in real time on a laptop CPU

# 

# All hyperparameters are in `anomaly\_detection/utils/config.py`.

# 

# \## Security design and limits

# 

# | Measure | Protects against | Limit |

# |---|---|---|

# | IP allowlist proxy (port 5020) | unknown hosts sending Modbus commands | IPs can be spoofed on a local network, and Modbus itself has no authentication or encryption |

# | Fernet (AES-128-CBC + HMAC-SHA256) for CSV transfer | reading or tampering with recorded data in transit | the key is shared manually (USB), with no key rotation |

# | Connection log | silent access | the sample log in `docs/` only shows granted connections, and rejection was not tested in it |

# 

# The live detector currently connects directly to port 502, so it does not go through the proxy.

# 

# \## Project structure

# 

# ```

# acquisition/        Modbus data recording

# anomaly\_detection/  LSTM training, live diagnosis, MLP baseline

# data/raw/           recorded state sequences (CSV)

# docs/               sample security log

# factoryio/          Factory I/O scene file

# security/           allowlist proxy, client, encrypted CSV transfer

# ```

# 

# \## Run it

# 

# ```bash

# pip install -r requirements.txt

# cd anomaly\_detection

# python train\_model.py          # trains the LSTM and saves the weights and the curves

# python baseline\_mlp.py         # MLP baseline

# python online\_diagnosis.py     # needs Factory I/O running with the Modbus TCP server

# ```

# 

# IP addresses for the lab setup are set at the top of each script.

# 

# \## Credits

# 

# \- \*\*Majd Al Ramadan:\*\* data acquisition, LSTM model and live diagnosis, security pipeline.

# \- \*\*Ange-Patrick:\*\* supervised the deep learning side of the project. He introduced the machine learning and deep learning foundations, explained the modelling choices, and guided the approach that Majd then developed and implemented.

# \- The MLP baseline architecture was designed by the internship tutor at CReSTIC.



