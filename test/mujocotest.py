import time
import mujoco
import mujoco.viewer

MODEL_XML = """
<mujoco>
    <option gravity="0 0 -9.81"/>
    <worldbody>
        <body name="ball" pos="0 0 1">
            <geom type="sphere" size="0.05" rgba="1 0 0 1"/>
            <joint type="free"/>
        </body>
    </worldbody>
</mujoco>
"""

def main():
    # Load model
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()