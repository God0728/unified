python scripts/sample.py \
    --checkpoint checkpoints/0428_v2/checkpoint_best.pt\
    --mode chain \
    --start="-7.95837e-17, -6.65266e-18,  6.93889e-17,-6.07421e-17,-0.17, 2.35922e-16,0.415559 ,0.264454, 0.971211,0.425521,-0.424698,0.924157,1,1,0,0" \
    --goal="1.38302e-13 ,-8.90602e-14 , 4.34791e-14, 0.270344, -0.158794, 0.0681215, 0.649719, 0.263705, 0.972901, 0.764773, -0.441853,  0.925739, 1, 0, 1 ,1" \
    --num_transitions 3 \
    --num_samples 10 \
    --guidance_scale 4.0 \
    --output outputs/seiko_chain_0428_test3.json

success 
