# Solid Modeling Basics

Original short reference notes authored for 3di-men's test suite -- not a
reproduction of any third-party textbook. Used to exercise the knowledge/
ingestion and retrieval pipeline end to end.

## Extrusion

Extrusion is defined as sweeping a closed 2D profile along a straight axis
by a given height to produce a solid. A rectangular profile extruded along
its normal produces a box; an extruded circular profile produces a
cylinder. The profile must be a closed, non-self-intersecting outline or
the resulting solid is degenerate.

## Fillets and Chamfers

A fillet is a rounded transition applied to a sharp edge, defined by a
radius. A chamfer is a flat, angled transition instead of a rounded one,
defined by a distance (or a distance and an angle). As a general rule of
thumb, fillet and chamfer sizes must be smaller than the thinnest adjacent
wall, or the operation can self-intersect and fail. A common failure mode
is requesting a fillet radius larger than half the shortest edge it sits
on -- this frequently causes the fillet operation to produce a degenerate
or self-intersecting surface. To avoid this, keep the fillet radius below
roughly one third of the adjacent wall thickness.

## Boolean Operations

Boolean union combines two solids into one, keeping all space occupied by
either input. Boolean difference subtracts the second solid from the
first, the standard way to cut a hole or pocket. Boolean intersection
keeps only the space occupied by both inputs. For example, a bolt hole
through a plate is typically modeled as a boolean difference between the
plate and a cylinder.

## Shelling

Shelling hollows out a solid, leaving a wall of a specified thickness and
removing one or more faces to create an opening. Shell wall thickness must
be strictly less than half the solid's smallest cross-sectional dimension,
or the inner offset surface self-intersects and the operation fails.
