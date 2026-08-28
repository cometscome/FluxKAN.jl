# Static reference fixture generated from the `b_splines` and `forward`
# equations in EfficientKAN at commit 7b6ce1c87f18c8bc90c208f6b494042344216b11.
# The values were evaluated independently in feature-last layout and then
# transposed to FluxKAN's documented feature-first layout. Keeping the expected
# numbers static ensures this test does not reuse FluxKAN's basis evaluator.
const EFFICIENTKAN_REFERENCE = (
    spline_order=2,
    grid_size=3,
    grid=Float64[
        -7 / 3 -5 / 3 -1 -1 / 3 1 / 3 1 5 / 3 7 / 3
        -7 / 3 -5 / 3 -1 -1 / 3 1 / 3 1 5 / 3 7 / 3
    ],
    input=Float64[
        -0.75 0.0 0.8
        -0.2 0.45 1.0
    ],
    basis=Float64[
        0.1953125 0.0 0.0
        0.734375 0.125 0.0
        0.0703125 0.75 0.045
        0.0 0.125 0.71
        0.0 0.0 0.245
        0.0 0.0 0.0
        0.32 0.0 0.0
        0.66 0.3403125 0.0
        0.02 0.644375 0.5
        0.0 0.0153125 0.5
    ],
    base_weight=Float64[
        0.3 -0.4
        -0.2 0.5
    ],
    spline_weight=Float64[
        0.1 -0.2 0.3 -0.4 0.5 -0.5 0.4 -0.3 0.2 -0.1
        0.25 0.0 -0.25 0.5 -0.5 0.3 -0.1 0.2 -0.4 0.6
    ],
    spline_scaler=Float64[
        1.2 0.8
        0.7 1.1
    ],
    output=Float64[
        -0.216471512470535 0.090284937889140 -0.264429555981375
        0.126181594854939 -0.148656172361425 0.520008372334584
    ],
)
