from flask import Flask, request, jsonify, send_file, render_template, Response
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for matplotlib
import os
import contextlib
import tempfile
import time
import threading
import json
import sys
import traceback

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial import HalfspaceIntersection, ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path


def create_bounding_box(mesh, z_offset):
    bounding_box = (np.max(mesh.vertices, axis=0) - np.min(mesh.vertices, axis=0)) + 30
    bounding_box_rounded_up = np.ceil(bounding_box)
    print("Bounding Box = ",bounding_box_rounded_up)
    l, b, h = bounding_box

    # Define the 8 vertices (bottom and top faces)
    vertices = np.array([
        [-l/2, -b/2, z_offset],         # 0
        [-l/2,  b/2, z_offset],         # 1
        [ l/2,  b/2, z_offset],         # 2
        [ l/2, -b/2, z_offset],         # 3
        [-l/2, -b/2, h + z_offset],     # 4
        [-l/2,  b/2, h + z_offset],     # 5
        [ l/2,  b/2, h + z_offset],     # 6
        [ l/2, -b/2, h + z_offset],     # 7
    ])

    # Define faces using the indices of vertices
    faces = np.array([
        [0, 1, 2], [0, 2, 3],         # bottom face
        [4, 5, 6], [4, 6, 7],         # top face
        [0, 1, 5], [0, 5, 4],         # left face
        [1, 2, 6], [1, 6, 5],         # front face
        [2, 3, 7], [2, 7, 6],         # right face
        [3, 0, 4], [3, 4, 7],         # back face
    ])

    # Create the mesh
    box_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    return box_mesh
def generate_gcode(cleaned_dict, bounding_box, output_filename, rotation_factor, z_offset, x_margin):
    """
    Generate GCODE from the cleaned trajectory dictionary.
    
    Parameters:
      cleaned_dict (dict): A nested dictionary (output from trajectory_cleanup) where
                           each key is a theta (in degrees) and its value is a dict that includes
                           "cleaned_convex_hull_points", a numpy array of shape (N,2) with columns [x, z].
      home (float): The home position for axes X and Y (e.g. +home).
      output_filename (str): Name of the file to write the GCODE.
      
    Behavior for each theta:
      1. Output a command to rotate axis B to the theta value.
      2. Retrieve the cleaned convex hull points.
         - If the first point's x is not negative, reverse the order.
      3. For each point in the cleaned convex hull points, output:
             G1 G90 G93 F600 X[pt_x] Y[pt_x] Z[pt_z] A[pt_z] B[theta]
      4. After cutting along all points for that theta, output:
             G1 G90 G93 F600 X[home] Y[home] Z[0] A[0] B[theta]
    
    The function writes all GCODE commands to the specified output file.
    """
    lines = []
    
    vertices = bounding_box.vertices
    
    xmax = np.max(vertices[:, 0])+x_margin
    xmin = np.min(vertices[:, 0])-x_margin
    zmax = np.max(vertices[:, 2])
    zmin = np.min(vertices[:, 2])
    
    lines.append("G21\nG17\nG92 B0")
    linearspeed = 1000
    rotaryspeed  = 1200

    line = f"G1 G90 G93 F{linearspeed} X{xmin:.3f} Y{xmin:.3f} Z{zmax:.3f} A{zmax:.3f}"
    lines.append(line)
    line = f"G1 G90 G93 F{linearspeed} X{xmin:.3f} Y{xmin:.3f} Z{zmin:.3f} A{zmin:.3f}"
    lines.append(line)
    
    flip = False
    # Iterate over theta keys in sorted order (optional)
    for theta_raw in sorted(cleaned_dict.keys()):
        theta = theta_raw*rotation_factor/360
        
        mesh = bounding_box.copy()
        
        # Convert theta to radians
        theta_rad = np.deg2rad(theta_raw)
        # Create a rotation matrix about the Z-axis
        R = trimesh.transformations.rotation_matrix(theta_rad, [0, 0, 1])
        # Apply the transformation
        mesh.apply_transform(R)
        vertices = mesh.vertices
        
        xmax = np.max(vertices[:, 0])
        xmin = np.min(vertices[:, 0])
        zmax = np.max(vertices[:, 2])
        zmin = np.min(vertices[:, 2])
        
        if flip == False:
            
            home = xmax + x_margin
            
            data = cleaned_dict[theta_raw]
            points = data["cleaned_convex_hull_points"]  # shape (N, 2): columns are [x, z]
            
            # Ensure that the sequence starts with a negative x value.
            # Check the first point; if x is not negative, reverse the order.
            if points[0, 0] >= 0:
                points = points[::-1]
            
            # 1. Rotate axis B to theta:
            # (Assume that G90 and G93 set absolute positioning and inverse time feed rate.)

            line = f"G1 G90 G93 F{rotaryspeed} B{theta:.3f}"
            lines.append(line)

            
            # 2. For each point, move X, Y to point's x; move Z, A to point's z; keep B constant.
            for pt in points:
                x_val, z_val = pt  # x and z in the rotated frame.
                z_val += z_offset
                # Here, both X and Y are set to the same value x_val,
                # and both Z and A are set to z_val.
                line = f"G1 G90 G93 F{linearspeed} X{x_val:.3f} Y{x_val:.3f} Z{z_val:.3f} A{z_val:.3f}"
                lines.append(line)

            # 3. After cutting, return to home: move X and Y to +home, Z and A to 0, B remains at theta.
            line = f"G1 G90 G93 F{linearspeed} X{home:.3f} Y{home:.3f} Z{zmin} A{zmin}"
            lines.append(line)
        else:
            home = xmin - x_margin
            data = cleaned_dict[theta_raw]
            points = data["cleaned_convex_hull_points"]  # shape (N, 2): columns are [x, z]
            
            # Ensure that the sequence starts with a negative x value.
            # Check the first point; if x is not negative, reverse the order.
            if points[0, 0] <= 0:
                points = points[::-1]
            
            # 1. Rotate axis B to theta:
            # (Assume that G90 and G93 set absolute positioning and inverse time feed rate.)

            line = f"G1 G90 G93 F{rotaryspeed} B{theta:.3f}"
            lines.append(line)

            
            # 2. For each point, move X, Y to point's x; move Z, A to point's z; keep B constant.
            for pt in points:
                x_val, z_val = pt  # x and z in the rotated frame.
                z_val += z_offset
                # Here, both X and Y are set to the same value x_val,
                # and both Z and A are set to z_val.
                line = f"G1 G90 G93 F{linearspeed} X{x_val:.3f} Y{x_val:.3f} Z{z_val:.3f} A{z_val:.3f}"
                lines.append(line)

            # 3. After cutting, return to home: move X and Y to +home, Z and A to 0, B remains at theta.
            line = f"G1 G90 G93 F{linearspeed} X{home:.3f} Y{home:.3f} Z{zmin} A{zmin}"
            lines.append(line)
        flip = not flip
    
    if not flip:
        xmax += x_margin
        line = f"G1 G90 G93 F{linearspeed} X{xmax:.3f} Y{xmax:.3f} Z{zmin:.3f} A{zmin:.3f}"
        lines.append(line)
    else:
        xmin -= x_margin
        line = f"G1 G90 G93 F{linearspeed} X{xmin:.3f} Y{xmin:.3f} Z{zmin:.3f} A{zmin:.3f}"
        lines.append(line)
    
    
    # Write the lines to the output file.
    with open(output_filename, "w") as f:
        for l in lines:
            f.write(l + "\n")
    
    print(f"GCODE generated and saved to {output_filename}")
def remove_duplicate_points(points, tol=1e-6):
    """
    Remove duplicate points (within tolerance) from a sequence while preserving order.
    
    Parameters:
        points (np.ndarray): Array of shape (N, 2) representing [x, z] coordinates.
        tol (float): Tolerance for considering two points as duplicates.
        
    Returns:
        np.ndarray: Array with duplicate points removed.
    """
    if len(points) == 0:
        return points
    
    unique = [points[0]]
    for p in points[1:]:
        if not np.any(np.linalg.norm(p - np.array(unique), axis=1) < tol):
            unique.append(p)
    return np.array(unique)
def remove_redundant_points(points, tol=1e-6):
    """
    Given an array of points of shape (N,2) in cyclic order, first remove duplicate points,
    then remove points that lie on the line joining their adjacent points.
    
    Parameters:
        points (np.ndarray): Array of shape (N,2) representing [x, z] coordinates.
        tol (float): Tolerance below which the area of the triangle is considered 0.
    
    Returns:
        np.ndarray: The cleaned array of points with duplicate and redundant points removed.
    """
    # Remove duplicate points first.
    points = remove_duplicate_points(points, tol)
    
    if len(points) < 3:
        return points
    
    cleaned = []
    N = len(points)
    for i in range(N):
        prev = points[i - 1]
        curr = points[i]
        next_pt = points[(i + 1) % N]
        # Compute the absolute area of the triangle (prev, curr, next_pt)
        area = 0.5 * np.abs(np.cross(curr - prev, next_pt - curr))
        if area > tol:
            cleaned.append(curr)
    return np.array(cleaned)
def trajectory_cleanup(reduced_dict, linear_precision=1):
    """
    Given a nested dictionary (from trajectory_optimizer) where each key is a theta (in degrees)
    and its value is a dict with at least:
         {"convex_hull": <ConvexHull object computed on X-Z points>, ... },
    this function:
      1. For each convex hull, extracts its boundary vertices (using .vertices attribute),
         then rounds these points to the nearest multiple of linear_precision.
      2. Finds the global minimum z value across all rounded convex hull boundary points.
      3. Adjusts (raises or lowers) the points for each theta such that the global minimum z becomes 0.
      4. Removes redundant points (points that lie along the line joining their neighbors).
      5. Stores the cleaned points in each dictionary entry under the key 
         "cleaned_convex_hull_points".
         
    Parameters:
        reduced_dict (dict): The dictionary output from trajectory_optimizer.
        linear_precision (float): The precision to which convex hull points are rounded.
                                  For example, if 0.5 is given, coordinates are rounded to the nearest 0.5.
        
    Returns:
        tuple: (modified_dict, global_min_z) where modified_dict is the input dictionary with 
               added keys "adjusted_convex_hull_points" and "cleaned_convex_hull_points" for 
               each theta, and global_min_z is the amount subtracted from each z coordinate.
    """
    global_min_z = np.inf
    global_min_x = np.inf

    # First pass: For each theta, extract the convex hull boundary points and round them.
    for theta, data in reduced_dict.items():
        hull = data["convex_hull"]
        # Extract the boundary points in order using the indices from hull.vertices.
        boundary_points = hull.points[hull.vertices]
        # Ensure cyclicity by appending the first point at the end (if desired).
        boundary_points = np.vstack([boundary_points, boundary_points[0]])
        # Round the points to the nearest multiple of linear_precision.
        rounded_points = np.round(boundary_points / linear_precision) * linear_precision
        data["rounded_convex_hull_points"] = rounded_points
        
        # Determine the minimum z from these rounded points.
        min_z = np.min(rounded_points[:, 1])
        if min_z < global_min_z:
            global_min_z = min_z


    # Second pass: Shift all rounded points so that global_min_z becomes 0 and remove redundant points.
    for theta, data in reduced_dict.items():
        rounded_points = data["rounded_convex_hull_points"]
        adjusted_points = rounded_points.copy()
        # Shift the z coordinates by subtracting global_min_z.
        adjusted_points[:, 1] -= global_min_z

        data["adjusted_convex_hull_points"] = adjusted_points
        
        # Remove redundant (collinear) points.
        cleaned_points = remove_redundant_points(adjusted_points)
        
        while True:
            if ((cleaned_points[0,1] == 0) and (cleaned_points[1,1] != 0)):
                break
            cleaned_points = np.roll(cleaned_points,1,0)
            
        data["cleaned_convex_hull_points"] = cleaned_points

    return reduced_dict
def generate_cut_obj(plane_dict, filename, feasible_point=None):
    """
    Given a nested dictionary (from trajectory_optimizer) where each key is a theta 
    (in degrees) and its value is a dict with:
         {"convex_hull": <ConvexHull>, "planes": <np.ndarray of shape (N,4)>},
    this function:
      1. Concatenates all the planes into one master array.
      2. Computes the intersection of the half-spaces defined by these planes.
      3. Computes the convex hull of the intersection points.
      4. Creates a trimesh mesh from the convex hull and exports it as an OBJ file.
    
    Parameters:
        plane_dict (dict): Optimized nested dictionary of cutting planes.
        filename (str): Output filename for the OBJ file.
        feasible_point (np.ndarray or None): A point inside the intersection.
            If None, defaults to [0,0,0] (ensure this lies inside your intersection).
    
    Returns:
        trimesh.Trimesh: The mesh representing the cut object.
    """
    # Concatenate all plane equations
    master_planes = np.vstack([data["planes"] for data in plane_dict.values()])
    
    # If no feasible point is provided, default to the origin.
    if feasible_point is None:
        feasible_point = np.array([0, 0, 0])
    
    # Compute the intersection of half-spaces.
    try:
        hs = HalfspaceIntersection(master_planes, feasible_point)
    except Exception as e:
        raise RuntimeError("Failed to compute halfspace intersection. "
                           "Ensure the feasible_point lies within the intersection. "
                           f"Original error: {e}")
    
    intersections = hs.intersections  # vertices of the intersection region
    
    # Compute the convex hull of these intersection points.
    hull = ConvexHull(intersections)
    
    # Create a mesh from the convex hull.
    cut_mesh = trimesh.Trimesh(vertices=intersections, faces=hull.simplices, process=True)
    
    # Export the mesh to an OBJ file.
    cut_mesh.export(filename)
    print(f"Cut object exported to {filename}")
    
    return cut_mesh
def compute_volume_from_planes(planes, feasible_point=None):
    """
    Given an array of planes (each row [a, b, c, d] for a*x+b*y+c*z+d <= 0),
    compute the volume of their intersection region.
    
    Parameters:
        planes (np.ndarray): Array of shape (n, 4) defining the half-spaces.
        feasible_point (np.ndarray or None): A point guaranteed to be inside the intersection.
                                             If None, [0,0,0] is used.
    
    Returns:
        float: Volume of the convex polyhedron.
    """
    if feasible_point is None:
        feasible_point = np.array([0, 0, 0])
    
    # Compute the intersection of half-spaces.
    try:
        hs = HalfspaceIntersection(planes, feasible_point)
    except Exception as e:
        print("HalfspaceIntersection failed with default options, trying with 'QJ' option.")
        hs = HalfspaceIntersection(planes, feasible_point, qhull_options="QJ")
    points = hs.intersections  # vertices of the intersection
    hull = ConvexHull(points)
    return hull.volume
def trajectory_optimizer(plane_dict, threshold_percentage, skip_percentage = 0.001):
    """
    Optimize the trajectory by dropping unnecessary theta values.
    
    Given a nested dictionary (from generate_cutting_planes) where each key is a theta (in degrees)
    and its value is a dict with:
      { "convex_hull": <ConvexHull>, "planes": <np.ndarray of shape (N,4)> },
    this function:
      1. Computes total_volume as the volume of the convex polyhedron formed by 
         all planes.
      2. Then, in a loop, while the relative volume difference 
             ((modified_volume/total_volume) - 1)*100 
         is below threshold_percentage, it determines which theta, if dropped,
         results in the smallest difference in volume from total_volume.
      3. It removes that theta (and its corresponding data) from the dictionary.
      4. If the removal causes the relative difference to exceed the threshold,
         the last removal is reverted, and the loop stops.
    
    Parameters:
        plane_dict (dict): Nested dictionary of cutting planes.
        threshold_percentage (float): Threshold percentage for relative volume difference.
    
    Returns:
        dict: The optimized dictionary (with some theta entries dropped).
    """
    # Work on a copy to avoid modifying the original dictionary.
    current_dict = plane_dict.copy()
    
    # Compute total_volume using all planes from the current dictionary.
    all_planes = np.vstack([data["planes"] for data in current_dict.values()])
    total_volume = compute_volume_from_planes(all_planes)
    print(f"Total volume using all planes: {total_volume:.4f}")
    
    while True:
        # Combine the remaining planes.
        all_planes_current = np.vstack([data["planes"] for data in current_dict.values()])
        modified_volume = compute_volume_from_planes(all_planes_current)
        # Compute relative difference as a percentage.
        relative_diff = ((modified_volume / total_volume) - 1) * 100
        #print(f"Current volume: {modified_volume:.4f}, relative diff: {relative_diff:.4f}%")
        
        # If the relative difference has reached or exceeded the threshold, stop.
        if relative_diff >= threshold_percentage:
            print("Threshold reached. Optimization complete.")
            break
        
        # Make a backup copy of the current dictionary.
        backup_dict = current_dict.copy()
        
        # For each theta in the dictionary, compute the volume difference if that theta is removed.
        differences = {}
        early_termination = False
        for theta in current_dict.keys():
            remaining_planes = np.vstack([data["planes"] 
                                          for key, data in current_dict.items() if key != theta])
            vol_if_removed = compute_volume_from_planes(remaining_planes)
            diff = abs((vol_if_removed / total_volume) - 1) * 100
            differences[theta] = diff
            if diff < skip_percentage:
                theta_to_remove = theta
                early_termination = True
                break
        
        if early_termination == False:
            # Select the theta whose removal minimizes the volume difference.
            theta_to_remove = min(differences, key=differences.get)
            
        # print(f"Removing theta {theta_to_remove}° minimizes volume difference to {differences[theta_to_remove]:.4f}%")
        
        # Remove that theta from the dictionary.
        del current_dict[theta_to_remove]
        
        # Recompute volume after removal.
        all_planes_current = np.vstack([data["planes"] for data in current_dict.values()])
        new_volume = compute_volume_from_planes(all_planes_current)
        new_rel_diff = ((new_volume / total_volume) - 1) * 100
        # print(f"After removal, new volume: {new_volume:.4f}, new relative diff: {new_rel_diff:.4f}%")
        
        
        # If the new relative difference exceeds the threshold, revert the last removal.
        if new_rel_diff >= threshold_percentage:
            current_dict = backup_dict  # revert
            print("Removal would exceed threshold; reverting last removal.")
            break
        
    return current_dict
def plot_volume_from_planes(plane_dict, theta=None, feasible_point=None):
    """
    Given a dictionary of cutting planes (structured as {theta: {"planes": np.ndarray}}),
    compute and plot the enclosed volume from their intersection.

    Parameters:
        plane_dict (dict): Dictionary of the form {theta: {"planes": np.ndarray (N,4)}}
                           where each row in the array is [a, b, c, d] defining a plane.
        theta (float or None): If specified, only the planes from this theta will be used.
                               If None, all planes across all thetas are used.
        feasible_point (np.ndarray or None): A point guaranteed to be inside the intersection.
                                             If None, the origin is assumed to be feasible.
    """
    # Extract planes based on the selected theta
    if theta is not None:
        if theta not in plane_dict:
            raise ValueError(f"Theta {theta} not found in plane dictionary.")
        planes = plane_dict[theta]["planes"]
    else:
        # Concatenate all planes from all theta values
        planes = np.vstack([data["planes"] for data in plane_dict.values()])

    if feasible_point is None:
        feasible_point = np.array([0, 0, 0])

    # Compute the intersection of half-spaces
    hs = HalfspaceIntersection(planes, feasible_point)
    points = hs.intersections  # The vertices of the intersection region

    # Compute the convex hull of these intersection points
    hull = ConvexHull(points)

    # Plotting the convex hull as a polyhedron
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a list of faces, where each face is a list of vertices.
    faces = [points[simplex] for simplex in hull.simplices]

    poly3d = Poly3DCollection(faces, facecolors='cyan', edgecolors='k', alpha=0.5)
    ax.add_collection3d(poly3d)

    # Scatter the vertices for clarity
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='r', s=50)

    # Set plot limits based on points
    max_range = (points.max(axis=0) - points.min(axis=0)).max() / 2.0
    mid = points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f"Enclosed Volume from Half-Spaces (Theta={theta if theta else 'All'})")
    plt.show()
def generate_cutting_planes(mesh, trajectories):
    """
    Generates a hierarchical dictionary where:
      - The first dimension is theta (used_theta).
      - The second dimension is the convex hull at that theta.
      - The third dimension is the array of planes corresponding to that convex hull.
    
    Parameters:
        mesh (trimesh.Trimesh): The original mesh.
        trajectories (dict): Dictionary where keys are theta values (in degrees)
                             and values are ConvexHull objects computed from the
                             projection (X-Z) of the rotated mesh.

    Returns:
        dict: A nested dictionary of the form:
              {
                  theta1: {
                      "convex_hull": ConvexHull object,
                      "planes": np.array of shape (N, 4) -> [[a, b, c, d], ...]
                  },
                  theta2: { ... },
                  ...
              }
    """
    results = {}

    # Loop over each theta and corresponding convex hull.
    for theta_deg, hull in trajectories.items():
        # Compute the rotation matrix about the Z-axis.
        theta_rad = np.deg2rad(theta_deg)
        R = trimesh.transformations.rotation_matrix(theta_rad, [0, 0, 1])

        # Rotate the mesh.
        rotated_mesh = mesh.copy()
        rotated_mesh.apply_transform(R)

        # Extract convex hull points in order.
        hull_indices = hull.vertices
        hull_pts = hull.points[hull_indices]

        # Close the hull loop.
        hull_pts = np.vstack([hull_pts, hull_pts[0]])

        # Store the plane equations for this theta.
        planes = []

        # Compute planes for consecutive pairs of points.
        for i in range(len(hull_pts) - 1):
            p1, p2 = hull_pts[i], hull_pts[i+1]

            # Compute normal in 2D (x-z plane).
            a_2d = p2[1] - p1[1]  # z2 - z1
            c_2d = p1[0] - p2[0]  # x1 - x2
            d = - (a_2d * p1[0] + c_2d * p1[1])

            # In the rotated frame, the plane equation is [a, 0, c, d]
            plane_rotated = np.array([a_2d, 0.0, c_2d, d])

            # Rotate the normal back to the original frame.
            n_rotated = plane_rotated[:3]
            n_original = R[:3, :3] @ n_rotated

            # Store the transformed plane equation.
            planes.append(np.hstack([n_original, plane_rotated[3]]))

        # Store the results for this theta.
        results[theta_deg] = {
            "convex_hull": hull,
            "planes": np.array(planes)
        }

    return results
def generate_trajectories(mesh, used_thetas):
    """
    For each theta in used_thetas (in degrees), rotate the mesh by that theta
    about the Z-axis, project the rotated mesh onto the X-Z plane, and compute
    the convex hull of the projection.
    
    Parameters:
        mesh (trimesh.Trimesh): The input 3D mesh.
        used_thetas (array-like): Array of theta values (in degrees) that were used.
    
    Returns:
        dict: A dictionary where keys are theta values (degrees) and values are the
              corresponding ConvexHull objects computed from the projected vertices.
    """
    trajectories = {}
    
    for theta in used_thetas:
        # Convert theta to radians
        theta_rad = np.deg2rad(theta)
        # Create a rotation matrix about the Z-axis
        R = trimesh.transformations.rotation_matrix(theta_rad, [0, 0, 1])
        # Make a copy of the mesh and apply the transformation
        rotated_mesh = mesh.copy()
        rotated_mesh.apply_transform(R)
        
        # Project the rotated mesh onto the X-Z plane:
        # Extract x and z coordinates (indices 0 and 2)
        projected_points = rotated_mesh.vertices[:, [0, 2]]
        
        # Compute the convex hull of the projected points
        hull = ConvexHull(projected_points)
        
        # Save the convex hull in the dictionary using the theta value as key.
        trajectories[theta] = hull
    
    return trajectories
def create_prism(n_sides, height=1.0, radius=1.0):
    """
    Creates a prism with a regular n-gon as the base.

    Parameters:
        n_sides (int): Number of sides for the base polygon.
        height (float): Height of the prism.
        radius (float): Radius of the circumcircle of the base polygon.
    
    Returns:
        trimesh.Trimesh: The generated prism mesh.
    """
    # Compute the angles for the base polygon vertices.
    angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
    # Bottom vertices (z = 0)
    bottom = np.stack((radius * np.cos(angles),
                       radius * np.sin(angles),
                       np.zeros(n_sides)), axis=1)
    # Top vertices (z = height)
    top = bottom.copy()
    top[:, 2] = height

    # Compute centers for bottom and top (for fan triangulation)
    center_bottom = np.array([0, 0, 0])
    center_top = np.array([0, 0, height])

    # Assemble vertices:
    #  - Bottom vertices: indices 0 to n_sides-1
    #  - Top vertices: indices n_sides to 2*n_sides-1
    #  - Bottom center: index 2*n_sides
    #  - Top center: index 2*n_sides + 1
    vertices = np.vstack([bottom, top, center_bottom, center_top])
    idx_center_bottom = 2 * n_sides
    idx_center_top = 2 * n_sides + 1

    faces = []
    # Triangulate bottom face (fan from bottom center)
    for i in range(n_sides):
        j = (i + 1) % n_sides
        faces.append([idx_center_bottom, i, j])
    # Triangulate top face (fan from top center)
    # Reverse order so that normals point upward
    for i in range(n_sides):
        j = (i + 1) % n_sides
        faces.append([idx_center_top, n_sides + j, n_sides + i])
    # Create side faces: each side is a quadrilateral split into 2 triangles.
    for i in range(n_sides):
        j = (i + 1) % n_sides
        # Vertices: bottom[i], bottom[j], top[j], top[i]
        faces.append([i, j, n_sides + j])
        faces.append([i, n_sides + j, n_sides + i])
    
    # Create and return the mesh
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
def compute_area_vs_theta(mesh, step, threshold):
    """
    For theta values from 0 to 180 degrees in increments of 'step',
    rotate the mesh by theta degrees about the Z-axis and compute the 
    sum of areas of all triangles whose rotated normal's x component 
    (via arctan2) is less than (threshold/2).
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        step (float): Step size in degrees for theta.
        threshold (float): Threshold (in radians) for the computed angle.
    
    Returns:
        thetas (np.ndarray): Array of theta values (in degrees).
        total_areas (np.ndarray): Array of total area values for each theta.
    """
    thetas = np.arange(0, 180, step)
    total_areas = []
    
    for theta_deg in thetas:
        theta_rad = np.deg2rad(theta_deg)
        R = trimesh.transformations.rotation_matrix(theta_rad, [0, 0, 1])
        R_mat = R[:3, :3]
        rotated_normals = (R_mat @ mesh.face_normals.T).T
        angles = np.abs(np.arctan2(rotated_normals[:, 1], rotated_normals[:, 0]))
        mask = (angles < (threshold / 2)) | (angles > (np.pi-(threshold / 2))) 
        area_sum = np.sum(mesh.area_faces[mask])
        total_areas.append(area_sum)
    
    return thetas, np.array(total_areas)
def mark_triangles_by_sorted_theta(mesh, step, threshold):
    """
    Computes the area vs theta series, sorts the theta values in descending 
    order based on the total area, then iterates over the sorted theta values.
    
    For each theta, it rotates the face normals and marks the triangles (faces) 
    that satisfy the condition (|arctan2(y,x)| < threshold/2) if they are not 
    already marked (using a new attribute `is_cut`).
    
    It collects only the theta values that were used to mark new faces. The iteration 
    stops when all faces are marked. Finally, it returns the used theta values (sorted 
    in descending order) and the modified mesh.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        step (float): Step size in degrees for theta.
        threshold (float): Threshold (in radians) used in the arctan2 mask.
    
    Returns:
        used_thetas (np.ndarray): The array of theta values (in degrees) that were used.
        mesh (trimesh.Trimesh): The modified mesh with an attribute `is_cut` for faces.
    """
    # Compute area vs theta series
    thetas, areas = compute_area_vs_theta(mesh, step, threshold)
    
    # Sort indices in descending order by area
    sort_idx = np.argsort(areas)[::-1]
    sorted_thetas = thetas[sort_idx]
    # sorted_areas = areas[sort_idx]  # (not used further in this version)
    
    # Initialize a boolean array for each face to track if it has been cut.
    mesh.is_cut = np.zeros(len(mesh.faces), dtype=bool)
    
    used_thetas = []
    
    for theta in sorted_thetas:
        theta_rad = np.deg2rad(theta)
        R = trimesh.transformations.rotation_matrix(theta_rad, [0, 0, 1])
        R_mat = R[:3, :3]
        rotated_normals = (R_mat @ mesh.face_normals.T).T
        angles = np.abs(np.arctan2(rotated_normals[:, 1], rotated_normals[:, 0]))
        mask = ((angles < (threshold / 2)) | (angles > (np.pi-(threshold / 2)))) & (~mesh.is_cut)
        # Mark the faces that satisfy the condition and haven't been marked yet.
        mesh.is_cut[mask] = True
        
        # If this theta caused any new faces to be marked, store it.
        if np.any(mask):
            used_thetas.append(theta)
        
        # Stop if all faces are marked.
        if np.all(mesh.is_cut):
            break
            
    return np.array(used_thetas), mesh


if __name__ == '__main__':
    
    # input_path = Path("smallrudder.stl")
    # #input_path = Path("P38/WingFuselage2.stl")
    # output_path = input_path.with_suffix(".gcode")
    # # Load your mesh (update the path as needed)
    # mesh = trimesh.load(input_path)
    # #mesh = create_prism(6)
    # # Compute the geometric center (mean of all vertices)
    # geometric_center = (np.max(mesh.vertices, axis=0) + np.min(mesh.vertices, axis=0))*1/2 
    
    # # Subtract the geometric center from each vertex
    # mesh.vertices -= geometric_center
    # mesh.vertices = mesh.vertices*1.010
    # # Convert theta to radians
    # theta_rad = np.deg2rad(90)
    # # Create a rotation matrix about the Z-axis
    # R = trimesh.transformations.rotation_matrix(theta_rad, [0, 1, 0])
    # # Apply the transformation
    # mesh.apply_transform(R)
    # # Set parameters:
    # step = 0.01                             # in degrees
    # threshold_deg = 0.01                    # threshold in degrees
    # threshold = np.deg2rad(threshold_deg)   # convert to radians
    
    # used_thetas, mesh = mark_triangles_by_sorted_theta(mesh, step, threshold)

    
    # trajectories = generate_trajectories(mesh, used_thetas)
    
    # dict_out = generate_cutting_planes(mesh, trajectories)
        
    # plot_volume_from_planes(dict_out)
    
    # # Volume Error Threshold (%)
    # error_threshold = 1.5
    
    # reduced_dict_out = trajectory_optimizer(dict_out, error_threshold, skip_percentage=1)
    
    # reduced_dict_out = trajectory_cleanup(reduced_dict_out,linear_precision=3)
    
    # plot_volume_from_planes(reduced_dict_out)
    
    # print("Number of initial cuts:", len(dict_out))
    
    # print("Number of final cuts used:",len(reduced_dict_out))
    
    # generate_cut_obj(reduced_dict_out, "cut_output.obj")
    
    # z_offset = 20
    
    # bounding_box = create_bounding_box(mesh, z_offset)
    
    # generate_gcode(reduced_dict_out, bounding_box, output_path, rotation_factor=400, z_offset= z_offset, x_margin = 50)
    print("Hi")
  
    

# Additional imports required for processing
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial import HalfspaceIntersection, ConvexHull
from pathlib import Path
import io
import base64

app = Flask(__name__, static_folder=".", static_url_path="")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'stl'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store processing status
processing_status = {
    "status": "idle",
    "progress": 0,
    "log": [],
    "error": None
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def reset_status():
    processing_status["status"] = "idle"
    processing_status["progress"] = 0
    processing_status["log"] = []
    processing_status["error"] = None

def log_message(message):
    timestamp = time.strftime("%H:%M:%S")
    processing_status["log"].append(f"[{timestamp}] {message}")
    print(f"[{timestamp}] {message}")

def update_progress(progress, message=None):
    processing_status["progress"] = progress
    if message:
        log_message(message)

@app.route('/')
def index():
    return app.send_static_file('index.html')

# Modify the get_status endpoint
@app.route('/status')
def get_status():
    status_data = processing_status.copy()
    # Add the console output and image flags
    status_data["console_output"] = processing_status.get("console_output", "")
    status_data["initial_plot"] = "initial_plot" in processing_status
    status_data["optimized_plot"] = "optimized_plot" in processing_status
    return jsonify(status_data)

@app.route('/process', methods=['POST'])
def process_file():
    reset_status()
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    # Get parameters
    params = {
        'step': float(request.form.get('step', 0.01)),
        'threshold': float(request.form.get('threshold', 0.01)),
        'error_threshold': float(request.form.get('error_threshold', 1.5)),
        'z_offset': float(request.form.get('z_offset', 20)),
        'rotation_factor': float(request.form.get('rotation_factor', 400)),
        'x_margin': float(request.form.get('x_margin', 50))
    }
    
    # Save the file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Process in background
    threading.Thread(target=process_stl, args=(filepath, filename, params)).start()
    
    return jsonify({
        "status": "processing",
        "filename": filename
    })

def process_stl(filepath, filename, params):
    processing_status["status"] = "processing"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename.replace('.stl', '.gcode'))
    captured_output = io.StringIO()    
    
    try:
        with contextlib.redirect_stdout(captured_output), contextlib.redirect_stderr(captured_output):
            update_progress(5, f"Loading mesh from {filename}")
            mesh = trimesh.load(filepath)
            validate_mesh(mesh)
            update_progress(10, "Centering mesh")
            geometric_center = (np.max(mesh.vertices, axis=0) + np.min(mesh.vertices, axis=0)) * 0.5
            mesh.vertices -= geometric_center
            mesh.vertices = mesh.vertices * 1.010
            
            update_progress(15, "Rotating mesh")
            theta_rad = np.deg2rad(90)
            R = trimesh.transformations.rotation_matrix(theta_rad, [0, 1, 0])
            mesh.apply_transform(R)
            
            update_progress(20, "Setting parameters")
            step = params['step']
            threshold_deg = params['threshold']
            threshold = np.deg2rad(threshold_deg)
            
            update_progress(30, "Marking triangles by theta")
            used_thetas, mesh = mark_triangles_by_sorted_theta(mesh, step, threshold)
            
            update_progress(40, "Generating trajectories")
            trajectories = generate_trajectories(mesh, used_thetas)
            
            update_progress(50, "Generating cutting planes")
            dict_out = generate_cutting_planes(mesh, trajectories)
            
            # Generate initial plot AFTER creating dict_out
            update_progress(60, "Computing initial volume")
            fig = plt.figure()
            plot_volume_from_planes(dict_out)
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            plt.close()
            processing_status["initial_plot"] = base64.b64encode(img_buf.getvalue()).decode('utf-8')
            
            update_progress(70, "Optimizing trajectories")
            error_threshold = params['error_threshold']
            reduced_dict_out = trajectory_optimizer(dict_out, error_threshold, skip_percentage=1)
            
            # Generate optimized plot AFTER optimization
            update_progress(75, "Generating optimized visualization")
            fig = plt.figure()
            plot_volume_from_planes(reduced_dict_out)
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            plt.close()
            processing_status["optimized_plot"] = base64.b64encode(img_buf.getvalue()).decode('utf-8')
            
            update_progress(80, "Cleaning up trajectories")
            reduced_dict_out = trajectory_cleanup(reduced_dict_out, linear_precision=3)
            
            update_progress(85, "Creating output object")
            
            update_progress(90, "Creating bounding box")
            z_offset = params['z_offset']
            bounding_box = create_bounding_box(mesh, z_offset)
            
            update_progress(95, "Generating GCode")
            rotation_factor = params['rotation_factor']
            x_margin = params['x_margin']
            generate_gcode(reduced_dict_out, bounding_box, output_path, rotation_factor, z_offset, x_margin)
            
            log_message(f"Number of initial cuts: {len(dict_out)}")
            log_message(f"Number of final cuts used: {len(reduced_dict_out)}")
            log_message(f"GCode generated and saved to {output_path}")
            
            update_progress(100, "Processing complete!")
            processing_status["status"] = "complete"
            processing_status["console_output"] = captured_output.getvalue()

    except Exception as e:
        log_message(f"Error: {str(e)}")
        processing_status["status"] = "error"
        processing_status["error"] = str(e)
        traceback.print_exc()

@app.route('/download/<filename>')
def download_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404

@app.route('/preview/<filename>')
def preview_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            content = f.read()
        return content
    else:
        return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

@app.route('/get-image/<type>')
def get_image(type):
    if type == 'initial' and 'initial_plot' in processing_status:
        return Response(base64.b64decode(processing_status["initial_plot"])), 200, {'Content-Type': 'image/png'}
    elif type == 'optimized' and 'optimized_plot' in processing_status:
        return Response(base64.b64decode(processing_status["optimized_plot"])), 200, {'Content-Type': 'image/png'}
    return "Image not found", 404

# Add this helper function
def validate_mesh(mesh):
    """Ensure mesh has sufficient complexity for processing"""
    if len(mesh.vertices) < 4:
        raise ValueError("Mesh must have at least 4 vertices")
    if mesh.volume < 1e-6:
        raise ValueError("Mesh appears to be flat or degenerate")

