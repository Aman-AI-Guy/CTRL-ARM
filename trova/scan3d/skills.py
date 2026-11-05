"""Skills for the existing Trova task loop; deterministic orbit planning owns poses."""
from ..workflow import receipt,receipt_error


def result(name,ok,reason,task,**extra):
    from ..agent import _result
    return _result(name,ok,reason,task,**extra)


def start_scan(host,task,argument):
    if not task.get('selected'):raise ValueError('Select an object first.')
    book=host.scan_book
    target=str(task['selected']['track_id'])
    if book.active:
        current=book._session(None)
        if current.get('reconstruction_only') or str(current.get('orientation_track_id',current['object_id']))!=target:
            raise ValueError('Finish the other scan before scanning this target.')
    else:
        book.start(target)
        if host.vision.intrinsics:
            book._session(None)['intrinsics']=host.vision.intrinsics
            book._write_index(book._session(None))
    task['scan_session']=book.active
    task['phase']='scan_started'
    return result('start_scan',True,'Scan session opened; no movement.',task,session=book.public(book._session(None)),executed=False)


def plan(host,task,argument):
    if not getattr(host,'scan_book',None) or not host.scan_book.active:
        start_scan(host,task,'')
    session=host.scan_book._session(None)
    if session.get('needs_reselection') or str(session.get('orientation_track_id',session['object_id']))!=str((task.get('selected') or {}).get('track_id')):
        raise ValueError('Reselect the object in the current orientation before planning.')
    task['scan_session']=session['id']
    views=[v for v in session['views'] if v['orientation']==session['orientation']]
    anchors=session.setdefault('orbit_anchors',{})
    if session['orientation'] not in anchors:
        anchors[session['orientation']]=host.robot().get('joints')
        host.scan_book._write_index(session)
    task['orbit_anchor_joints']=anchors[session['orientation']]
    report=host.orbits.plan(host,task,views)
    task['scan_plan']=report
    next_=report.get('next')
    if not next_:
        task.update(phase='blocked',motion_lock=report['reason'])
        return result('plan_orbit',False,report['reason'],task,terminal=True,executed=False,scan_plan=report)
    task.update(planned_target=next_['target_mm'],planned_joints=next_['joints_deg'],
                orbit_geometry=report['geometry'],orbit_speed=next_['joint_speed_deg_s'],
                proposal=next_['gateway'],scan_viewpoint=next_['id'],phase='planned')
    return result('plan_orbit',True,'Joint trajectory proposed and checked; not executed.',task,scan_plan=report,executed=False)


def observe(host,task,argument):
    from ..agent import qualification_blockers,_selection_unreliable
    if not task.get('planned_joints'):
        planned=plan(host,task,'')
        if not planned['ok']:return planned
    if task.get('_confirmed'):
        invalid=receipt_error(host,task,task['_confirmed'])
        if invalid:raise ValueError(invalid)
    if task.get('motion_lock'):raise ValueError(task['motion_lock'])
    identity=_selection_unreliable(task)
    if identity:raise ValueError(identity)
    blockers=qualification_blockers(host,task)
    if blockers:
        task.update(phase='held',motion_lock='; '.join(blockers))
        return result('observe_from_viewpoint',False,task['motion_lock'],task,terminal=True,executed=False)
    geometry=host.orbits.context(host,task)
    if geometry.signature()!=task['orbit_geometry']:
        raise ValueError('Orbit geometry or measured settings changed; request a new proposal.')
    decision=host.gateway.authorize_joints(task['planned_joints'],task['orbit_speed'],source='scan_orbit',geometry=geometry)
    task['proposal']=decision.as_dict()
    if not decision.ok or not decision.allow_motion:
        task.update(phase='held',motion_lock=decision.reason)
        return result('observe_from_viewpoint',False,decision.reason,task,terminal=True,executed=False)
    if not task.get('_confirmed'):
        task['confirmation']=receipt(host,task)
        task['phase']='awaiting_confirmation'
        return result('observe_from_viewpoint',True,'Validated joint trajectory awaiting operator confirmation.',task,terminal=True,executed=False)
    if host.cancel.is_set():raise ValueError('STOP or cancellation requested.')
    host.active_task=task
    outcome=host.execute_orbit(task,geometry)
    task.pop('_confirmed',None);task.pop('confirmation',None)
    task['phase']='moved' if outcome['ok'] else 'blocked'
    if not outcome['ok']:task['motion_lock']=outcome['reason']
    task['orbit_reached']=bool(outcome['ok'])
    return result('observe_from_viewpoint',bool(outcome['ok']),outcome['reason'],task,
                  executed=bool(outcome.get('motion_observed') or outcome['ok']),
                  verified_tool_position=False,movement_status_unknown=outcome.get('movement_status_unknown',False))


def coverage(host,task,argument):
    session=host.scan_book._session(None)
    views=[v for v in session['views'] if v['orientation']==session['orientation']]
    good=[v for v in views if v.get('accepted') or v.get('reconstruction_accepted')]
    task['phase']='evaluated'
    task['coverage']={'accepted_views':len(good),'total_views':len(views),'surface_coverage_known':False,
                      'orientation':session['orientation'],'cross_orientation_registration':False}
    continuing=bool(task.get('_scan_continuation'))
    if continuing:
        budget=max(3,int(round(host.orbits.settings.span_deg/host.orbits.settings.step_deg))+1)
        task['coverage']['view_budget']=budget
        if len(good)>=budget:
            task['phase']='scan_ready_for_reconstruction'
    return result('evaluate_coverage',True,'Viewpoint coverage counted; surface completeness is unknown.',task,
                  coverage=task['coverage'],executed=False,terminal=not continuing)


def reconstruct(host,task,argument):
    session=host.scan_book._session(None)
    if task.get('phase')=='scan_ready_for_reconstruction': host.scan_book.finish()
    job=host.reconstruction.start(session)
    task['reconstruction_job']=job['id'];task['phase']='reported'
    return result('reconstruct_object',True,'Reconstruction queued; no result is claimed yet.',task,reconstruction=job,terminal=True,executed=False)


def inspect(host,task,argument):
    key=task.get('reconstruction_job')
    if key is None:
        jobs=list(host.reconstruction.jobs)
        if not jobs:raise ValueError('No reconstruction job exists.')
        key=jobs[-1]
    report=host.reconstruction.get(key)
    task['phase']='reported'
    return result('inspect_reconstruction',report['status']!='failed',report['reason'],task,reconstruction=report,terminal=True,executed=False)


def reorient(host,task,argument):
    task['phase']='reported'
    return result('request_object_reorientation',True,
        'No additional surface coverage is guaranteed. Cancel motion, park through a separately validated proposal, then turn the object and select it again. A new group is required; cross-group registration is unavailable.',
        task,terminal=True,executed=False,human_assistance_requested=True)


def finish(host,task,argument):
    report=host.scan_book.finish() if host.scan_book.active else None
    task['phase']='reported'
    return result('finish_task',True,'Acquisition finished; reconstruction is a separate operation.',task,session=report,terminal=True,executed=False)
