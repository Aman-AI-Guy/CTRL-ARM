import json
import pytest
import numpy as np
from trova.localization import make_router
from trova.geom import VisionStore
from trova.calibration_compare import V2


def test_experimental_v2_never_activates_metric_or_hardware(tmp_path):
    from test_localization_providers import robot
    if not V2.exists():pytest.skip('Optional V2 archive absent')
    router=make_router(VisionStore(),tmp_path)
    capture=json.loads((V2.parents[2]/'arm_pc/runs/20261105-044620_hover/hover_target.json').read_text())
    state=robot(capture['q0'])
    router.set_context({'kind':'pi','host':'camera','rotate_reported':180,'extra_rotate':0},state)
    selected={'track_id':1,'box':[0,0,1600,900], 'geometric_reference':{'kind':'known_height_point','pixel':capture['pixel'],'height_mm':55,'snapshot_id':'recorded_test'}}
    result=router.experimental_localize_observation(state,selected,[1600,900])
    np.testing.assert_allclose(result['point_mm'],capture['target_mm'],atol=1e-8)
    assert result['experimental'] and not result['metric_permitted'] and not result['calibration_valid']
    assert result['uncertainty_mm'] is None and not result['hardware_authorized']
    assert router.assess([1600,900])['active_provider']=='image_space'
    assert not router.localize_observation(state,selected,[1600,900])['ok']
    with pytest.raises(ValueError,match='restricted'):router.experimental_localize_observation(state,selected,[1600,900],mode='hardware')
    selected['box']=[0,0,1,1]
    with pytest.raises(ValueError,match='outside selected'):router.experimental_localize_observation(state,selected,[1600,900])
