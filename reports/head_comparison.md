# Brand head comparison (frozen DINOv2 ViT-S/14 features, 224 px)

Train 4305 crops (x2 with mirror), validation 788 crops (53 own Tashkent photos, 735 external). Same split for every head.

| Head | Val accuracy | Own photos | External | Macro-F1 | Train time |
|---|---|---|---|---|---|
| MLP head (ours) | 0.666 | 0.774 | 0.659 | 0.651 | 11s |
| Logistic regression | 0.494 | 0.717 | 0.478 | 0.478 | 2s |
| kNN (k=10, cosine) | 0.473 | 0.717 | 0.456 | 0.461 | 0s |
| XGBoost | 0.524 | 0.717 | 0.510 | 0.508 | 211s |
| CatBoost | 0.520 | 0.774 | 0.502 | 0.507 | 297s |
| LightGBM | 0.543 | 0.736 | 0.529 | 0.531 | 282s |

### MLP head (ours)

```
               precision    recall  f1-score   support

         AITO      0.821     0.852     0.836        27
         Aion      0.722     0.867     0.788        30
          BMW      0.964     0.900     0.931        30
          BYD      0.588     0.625     0.606        32
      Changan      0.567     0.708     0.630        24
        Chery      0.526     0.455     0.488        22
    Chevrolet      0.694     0.685     0.690        73
       Daewoo      0.485     0.640     0.552        25
       Deepal      0.842     0.800     0.821        20
     Dongfeng      0.594     0.679     0.633        28
        Exeed      0.676     0.885     0.767        26
        Geely      0.643     0.346     0.450        26
        Haval      0.538     0.350     0.424        20
       Hongqi      0.583     0.700     0.636        20
      Hyundai      0.579     0.379     0.458        29
          JAC      0.364     0.235     0.286        17
       Jaecoo      0.778     0.933     0.848        15
       Jetour      0.625     0.652     0.638        23
          Kia      0.550     0.367     0.440        30
         Lada      0.381     0.533     0.444        15
    Leapmotor      0.786     0.880     0.830        25
      Li Auto      0.833     0.962     0.893        26
Mercedes-Benz      0.667     0.514     0.581        35
       Nissan      0.571     0.522     0.545        23
        Omoda      0.710     0.815     0.759        27
         Tank      0.917     0.957     0.936        23
       Toyota      0.409     0.360     0.383        25
   Volkswagen      0.640     0.593     0.615        27
        Voyah      0.708     0.810     0.756        21
        Zeekr      0.815     0.917     0.863        24

     accuracy                          0.666       788
    macro avg      0.653     0.664     0.651       788
 weighted avg      0.661     0.666     0.657       788
```

### Logistic regression

```
               precision    recall  f1-score   support

         AITO      0.571     0.741     0.645        27
         Aion      0.704     0.633     0.667        30
          BMW      0.839     0.867     0.852        30
          BYD      0.414     0.375     0.393        32
      Changan      0.344     0.458     0.393        24
        Chery      0.294     0.227     0.256        22
    Chevrolet      0.833     0.411     0.550        73
       Daewoo      0.325     0.520     0.400        25
       Deepal      0.538     0.700     0.609        20
     Dongfeng      0.500     0.429     0.462        28
        Exeed      0.382     0.500     0.433        26
        Geely      0.286     0.231     0.255        26
        Haval      0.194     0.350     0.250        20
       Hongqi      0.480     0.600     0.533        20
      Hyundai      0.294     0.172     0.217        29
          JAC      0.154     0.118     0.133        17
       Jaecoo      0.583     0.933     0.718        15
       Jetour      0.278     0.217     0.244        23
          Kia      0.400     0.200     0.267        30
         Lada      0.265     0.600     0.367        15
    Leapmotor      0.559     0.760     0.644        25
      Li Auto      0.742     0.885     0.807        26
Mercedes-Benz      0.440     0.314     0.367        35
       Nissan      0.526     0.435     0.476        23
        Omoda      0.528     0.704     0.603        27
         Tank      0.769     0.870     0.816        23
       Toyota      0.333     0.200     0.250        25
   Volkswagen      0.588     0.370     0.455        27
        Voyah      0.520     0.619     0.565        21
        Zeekr      0.692     0.750     0.720        24

     accuracy                          0.494       788
    macro avg      0.479     0.506     0.478       788
 weighted avg      0.509     0.494     0.484       788
```

### kNN (k=10, cosine)

```
               precision    recall  f1-score   support

         AITO      0.400     0.370     0.385        27
         Aion      0.654     0.567     0.607        30
          BMW      0.800     0.800     0.800        30
          BYD      0.414     0.375     0.393        32
      Changan      0.311     0.583     0.406        24
        Chery      0.267     0.182     0.216        22
    Chevrolet      0.614     0.589     0.601        73
       Daewoo      0.423     0.440     0.431        25
       Deepal      0.500     0.750     0.600        20
     Dongfeng      0.333     0.286     0.308        28
        Exeed      0.415     0.654     0.507        26
        Geely      0.200     0.192     0.196        26
        Haval      0.345     0.500     0.408        20
       Hongqi      0.625     0.500     0.556        20
      Hyundai      0.182     0.138     0.157        29
          JAC      0.400     0.118     0.182        17
       Jaecoo      0.706     0.800     0.750        15
       Jetour      0.429     0.391     0.409        23
          Kia      0.346     0.300     0.321        30
         Lada      0.368     0.467     0.412        15
    Leapmotor      0.517     0.600     0.556        25
      Li Auto      0.586     0.654     0.618        26
Mercedes-Benz      0.484     0.429     0.455        35
       Nissan      0.471     0.348     0.400        23
        Omoda      0.700     0.519     0.596        27
         Tank      0.826     0.826     0.826        23
       Toyota      0.348     0.320     0.333        25
   Volkswagen      0.333     0.333     0.333        27
        Voyah      0.571     0.571     0.571        21
        Zeekr      0.481     0.542     0.510        24

     accuracy                          0.473       788
    macro avg      0.468     0.471     0.461       788
 weighted avg      0.475     0.473     0.468       788
```

### XGBoost

```
               precision    recall  f1-score   support

         AITO      0.615     0.593     0.604        27
         Aion      0.645     0.667     0.656        30
          BMW      0.852     0.767     0.807        30
          BYD      0.485     0.500     0.492        32
      Changan      0.531     0.708     0.607        24
        Chery      0.308     0.182     0.229        22
    Chevrolet      0.587     0.603     0.595        73
       Daewoo      0.368     0.560     0.444        25
       Deepal      0.583     0.700     0.636        20
     Dongfeng      0.486     0.643     0.554        28
        Exeed      0.433     0.500     0.464        26
        Geely      0.208     0.192     0.200        26
        Haval      0.368     0.350     0.359        20
       Hongqi      0.562     0.450     0.500        20
      Hyundai      0.318     0.241     0.275        29
          JAC      0.267     0.235     0.250        17
       Jaecoo      0.727     0.533     0.615        15
       Jetour      0.355     0.478     0.407        23
          Kia      0.500     0.267     0.348        30
         Lada      0.320     0.533     0.400        15
    Leapmotor      0.710     0.880     0.786        25
      Li Auto      0.657     0.885     0.754        26
Mercedes-Benz      0.375     0.343     0.358        35
       Nissan      0.522     0.522     0.522        23
        Omoda      0.714     0.556     0.625        27
         Tank      0.864     0.826     0.844        23
       Toyota      0.300     0.120     0.171        25
   Volkswagen      0.625     0.370     0.465        27
        Voyah      0.500     0.667     0.571        21
        Zeekr      0.680     0.708     0.694        24

     accuracy                          0.524       788
    macro avg      0.516     0.519     0.508       788
 weighted avg      0.523     0.524     0.515       788
```

### CatBoost

```
               precision    recall  f1-score   support

         AITO      0.484     0.556     0.517        27
         Aion      0.700     0.700     0.700        30
          BMW      0.897     0.867     0.881        30
          BYD      0.464     0.406     0.433        32
      Changan      0.410     0.667     0.508        24
        Chery      0.250     0.091     0.133        22
    Chevrolet      0.667     0.603     0.633        73
       Daewoo      0.382     0.520     0.441        25
       Deepal      0.682     0.750     0.714        20
     Dongfeng      0.516     0.571     0.542        28
        Exeed      0.405     0.654     0.500        26
        Geely      0.227     0.192     0.208        26
        Haval      0.250     0.250     0.250        20
       Hongqi      0.550     0.550     0.550        20
      Hyundai      0.238     0.172     0.200        29
          JAC      0.333     0.235     0.276        17
       Jaecoo      0.846     0.733     0.786        15
       Jetour      0.344     0.478     0.400        23
          Kia      0.280     0.233     0.255        30
         Lada      0.375     0.600     0.462        15
    Leapmotor      0.655     0.760     0.704        25
      Li Auto      0.679     0.731     0.704        26
Mercedes-Benz      0.448     0.371     0.406        35
       Nissan      0.455     0.435     0.444        23
        Omoda      0.800     0.593     0.681        27
         Tank      0.938     0.652     0.769        23
       Toyota      0.375     0.240     0.293        25
   Volkswagen      0.583     0.519     0.549        27
        Voyah      0.552     0.762     0.640        21
        Zeekr      0.615     0.667     0.640        24

     accuracy                          0.520       788
    macro avg      0.513     0.519     0.507       788
 weighted avg      0.523     0.520     0.514       788
```

### LightGBM

```
               precision    recall  f1-score   support

         AITO      0.621     0.667     0.643        27
         Aion      0.625     0.667     0.645        30
          BMW      0.867     0.867     0.867        30
          BYD      0.429     0.375     0.400        32
      Changan      0.486     0.708     0.576        24
        Chery      0.333     0.273     0.300        22
    Chevrolet      0.582     0.630     0.605        73
       Daewoo      0.382     0.520     0.441        25
       Deepal      0.750     0.750     0.750        20
     Dongfeng      0.533     0.571     0.552        28
        Exeed      0.436     0.654     0.523        26
        Geely      0.286     0.231     0.255        26
        Haval      0.278     0.250     0.263        20
       Hongqi      0.588     0.500     0.541        20
      Hyundai      0.333     0.241     0.280        29
          JAC      0.333     0.176     0.231        17
       Jaecoo      0.714     0.667     0.690        15
       Jetour      0.375     0.522     0.436        23
          Kia      0.429     0.300     0.353        30
         Lada      0.296     0.533     0.381        15
    Leapmotor      0.667     0.800     0.727        25
      Li Auto      0.700     0.808     0.750        26
Mercedes-Benz      0.500     0.371     0.426        35
       Nissan      0.600     0.522     0.558        23
        Omoda      0.783     0.667     0.720        27
         Tank      0.938     0.652     0.769        23
       Toyota      0.471     0.320     0.381        25
   Volkswagen      0.667     0.444     0.533        27
        Voyah      0.577     0.714     0.638        21
        Zeekr      0.643     0.750     0.692        24

     accuracy                          0.543       788
    macro avg      0.541     0.538     0.531       788
 weighted avg      0.546     0.543     0.537       788
```
