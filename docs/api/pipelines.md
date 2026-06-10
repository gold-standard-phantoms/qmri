# qmri.pipelines

End-to-end quantitative MRI workflows that combine the pure signal models in
`qmri` with the file handling in `qmri.io`. Pipelines are file-in / file-out:
they load images, run the fit, and write maps and reports.

Provided by the `qmri-pipelines` package.

## Multi-Echo Thermometry

::: qmri.pipelines.thermometry.multiecho
    options:
      members:
        - run_multiecho_thermometry
        - MultiEchoThermometryReport
