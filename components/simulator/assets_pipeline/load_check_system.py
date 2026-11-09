"""gz-sim Python system for load_test.py: records every link's start/end
position and peak speed, and every joint's position range, and writes them
as JSON to $LOAD_CHECK_OUT every 0.5 s of sim time (and on the last step)."""
import json
import math
import os


class LoadCheck:
    def configure(self, entity, sdf, ecm, event_mgr):
        from gz.sim import World
        self.world = World(entity)
        self.out = os.environ["LOAD_CHECK_OUT"]
        self.models = None
        self.data = {"t": 0.0, "nan": False, "models": {}}

    def _setup(self, ecm):
        from gz.sim import Joint, Link, Model
        self.models = {}
        for m in self.world.models(ecm):
            model = Model(m)
            name = model.name(ecm)
            if name == "ground_plane":
                continue
            links = {Link(e).name(ecm): Link(e) for e in model.links(ecm)}
            joints = {Joint(e).name(ecm): Joint(e) for e in model.joints(ecm)}
            for l in links.values():
                l.enable_velocity_checks(ecm, True)
            for j in joints.values():
                j.enable_position_check(ecm)
            self.models[name] = (links, joints)
            self.data["models"][name] = {"links": {}, "joints": {}}

    def post_update(self, info, ecm):
        if self.models is None:
            self._setup(ecm)
            return
        t = info.sim_time.total_seconds()
        for name, (links, joints) in self.models.items():
            d = self.data["models"][name]
            for ln, l in links.items():
                pose = l.world_pose(ecm)
                v = l.world_linear_velocity(ecm)
                if pose is None:
                    continue
                p = [pose.pos().x(), pose.pos().y(), pose.pos().z()]
                speed = math.sqrt(v.x() ** 2 + v.y() ** 2 + v.z() ** 2) if v is not None else 0.0
                if any(math.isnan(x) for x in p) or math.isnan(speed):
                    self.data["nan"] = True
                    continue
                r = d["links"].setdefault(ln, {"p0": p, "vmax": 0.0, "t_vmax": 0.0})
                r["p"] = p
                if speed > r["vmax"]:
                    r["vmax"], r["t_vmax"] = speed, t
            for jn, j in joints.items():
                q = j.position(ecm)
                if not q:
                    continue
                r = d["joints"].setdefault(jn, {"qmin": q[0], "qmax": q[0]})
                r["q"] = q[0]
                r["qmin"], r["qmax"] = min(r["qmin"], q[0]), max(r["qmax"], q[0])
        self.data["t"] = t
        if int(round(t * 1000)) % 500 == 0:
            self.write()

    def write(self):
        tmp = self.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f)
        os.replace(tmp, self.out)


def get_system():
    return LoadCheck()
