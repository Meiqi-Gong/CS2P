# CS2P: Cross-Scale Learning for Spectral Pansharpening

CS2P is a cross-scale learning framework for multispectral and hyperspectral pansharpening.

## Network Architecture

<p align="center">
  <img src="assets/overall_network.png" alt="CS2P network architecture" width="100%">
</p>

*Global Scale-Adaptive Network (GSAN).*

## Scale Domain Bridge (SDB)

<p align="center">
  <img src="assets/sdb.png" alt="Scale Domain Bridge" width="700">
</p>

*SDB progressively transfers full-resolution information to the reduced-resolution model.*

## Training Time

RRS: reduced-resolution scale; FRS: full-resolution scale.
Per-epoch times are measured on a single NVIDIA A800 80GB GPU. Total time assumes parallel RRS and FRS training on two GPUs, followed by SDB training.

<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th colspan="2">RRS training</th>
      <th colspan="2">FRS training</th>
      <th colspan="2">SDB training</th>
      <th rowspan="2">Total time (h)</th>
    </tr>
    <tr>
      <th>Epochs</th><th>min/epoch</th>
      <th>Epochs</th><th>min/epoch</th>
      <th>Epochs</th><th>min/epoch</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>QB</td><td>32</td><td>21</td><td>4</td><td>37</td><td>2</td><td>42</td><td>12.6</td></tr>
    <tr><td>GF2</td><td>40</td><td>21</td><td>5</td><td>37</td><td>5</td><td>42</td><td>17.5</td></tr>
    <tr><td>WV2</td><td>40</td><td>25</td><td>13</td><td>62</td><td>10</td><td>114</td><td>35.7</td></tr>
    <tr><td>Chikusei</td><td>35</td><td>38</td><td>15</td><td>71</td><td>10</td><td>115</td><td>41.3</td></tr>
    <tr><td>ColonCancer</td><td>26</td><td>115</td><td>10</td><td>177</td><td>8</td><td>180</td><td>73.8</td></tr>
  </tbody>
</table>

## Code Availability

The code will be released upon acceptance.
