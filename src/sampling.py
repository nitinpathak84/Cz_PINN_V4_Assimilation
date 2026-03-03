from physicsnemo.sym.geometry.geometry_dataloader import GeometryDatapipe

def make_volume_sampler(geom, npts: int, device, num_workers: int = 1):
    return GeometryDatapipe(
        geom_objects=[geom],
        batch_size=1,
        num_points=npts,
        sample_type="volume",
        device=device,
        num_workers=num_workers,
        requested_vars=["x", "y", "sdf"],
    )

def make_surface_sampler(geom, npts: int, device, num_workers: int = 1):
    return GeometryDatapipe(
        geom_objects=[geom],
        batch_size=1,
        num_points=npts,
        sample_type="surface",
        device=device,
        num_workers=num_workers,
        requested_vars=["x", "y"],
    )