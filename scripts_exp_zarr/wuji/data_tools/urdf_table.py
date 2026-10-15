"""Print the joint chain base -> palm of a URDF as a table."""
import sys
import xml.etree.ElementTree as ET

def main(path, tip):
    root = ET.parse(path).getroot()
    joints = {}
    for j in root.findall("joint"):
        child = j.find("child").get("link")
        joints[child] = j
    chain = []
    link = tip
    while link in joints:
        j = joints[link]
        chain.append(j)
        link = j.find("parent").get("link")
    print(f"base link: {link}")
    for j in reversed(chain):
        o = j.find("origin")
        a = j.find("axis")
        lim = j.find("limit")
        lim_s = "" if lim is None else f"lim=[{lim.get('lower')}, {lim.get('upper')}] vel={lim.get('velocity')} eff={lim.get('effort')}"
        print(f"{j.get('name'):22s} {j.get('type'):9s} {j.find('parent').get('link'):>22s} -> {j.find('child').get('link'):22s} "
              f"xyz={o.get('xyz') if o is not None else None} rpy={o.get('rpy') if o is not None else None} "
              f"axis={a.get('xyz') if a is not None else None} {lim_s}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
