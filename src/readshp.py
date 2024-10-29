import shapefile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import ConvexHull

# Read the shapefile
sf = shapefile.Reader('/opt/project/dataset/World-Administrative-Boundaries/world-administrative-boundaries.shp')
shapes = sf.shapes()[255:]
print(len(shapes))
# Create a figure and axis with a cartopy projection
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# Add coastlines and countries
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)

# Plot each shape as a filled polygon
for shape in shapes:
    points = shape.points
    
    hull = ConvexHull(points)
    hull_points = [points[vertex] for vertex in hull.vertices]
    polygon = patches.Polygon(hull_points, closed=True, edgecolor='red', facecolor='green', alpha=0.5, transform=ccrs.PlateCarree())
    ax.add_patch(polygon)

# Adjust the axis limits to fit all polygons
ax.set_global()

# Save the plot
plt.savefig('/opt/project/dataset/World-Administrative-Boundaries/world-administrative-boundaries2.png')
plt.show()